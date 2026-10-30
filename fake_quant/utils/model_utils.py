import torch
import typing
import transformers
import types
from . import utils
import os
import logging
import llava
from . import quant_utils
import torch.nn as nn
import torch.nn.functional as F


def _optional_transformers_class(path: str):
    obj = transformers
    try:
        for part in path.split("."):
            obj = getattr(obj, part)
    except AttributeError:
        return ()
    return obj


OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
OPT_LAYER = transformers.models.opt.modeling_opt.OPTDecoderLayer
LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer
LLAVA_MODEL = llava.model.LlavaLlamaForCausalLM
LLAVA_NEXT_HF = _optional_transformers_class("models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration")
LLAMAV_MODLE = _optional_transformers_class("models.mllama.modeling_mllama.MllamaForConditionalGeneration")
SMOVLM_MODEL = _optional_transformers_class("models.idefics3.modeling_idefics3.Idefics3ForConditionalGeneration")
#### QWEN model
QWEN25_MODEL = _optional_transformers_class("models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration")
QWEN2_MODEL = _optional_transformers_class("models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration")
QWEN3_MODEL = _optional_transformers_class("models.qwen3_vl.modeling_qwen3_vl.Qwen3VLForConditionalGeneration")
QWEN3_MODEL_TYPE = QWEN3_MODEL or "Qwen3VLForConditionalGeneration"
QWEN_VL_MODEL = 'qwen_vl_chat'
MOLMO_MODEL = 'molmo-o-model'
BLIP2_INS = _optional_transformers_class("models.instructblip.modeling_instructblip.InstructBlipForConditionalGeneration")
##### InternVL model
INTERNVL_MODEL = _optional_transformers_class("models.internvl.modeling_internvl.InternVLForConditionalGeneration")
def _is_qwen3_model(model):
    return (QWEN3_MODEL is not None and isinstance(model, QWEN3_MODEL)) or model.__class__.__name__ == "Qwen3VLForConditionalGeneration"


def is_molmo_model(model) -> bool:
    if model is None:
        return False
    class_name = model.__class__.__name__.lower()
    config = getattr(model, "config", None)
    config_name = str(getattr(config, "_name_or_path", "")).lower()
    model_type = str(getattr(config, "model_type", "")).lower()
    inner = getattr(model, "model", None)
    inner_name = inner.__class__.__name__.lower() if inner is not None else ""
    return (
        "molmo" in class_name
        or "molmo" in inner_name
        or "molmo" in config_name
        or "molmo" in model_type
    )


def _molmo_inner_model(model):
    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        return model.model
    if hasattr(model, "transformer"):
        return model
    raise AttributeError("Cannot locate Molmo inner model with transformer.")


def _molmo_transformer(model):
    return _molmo_inner_model(model).transformer


def _molmo_blocks(model):
    transformer = _molmo_transformer(model)
    if hasattr(transformer, "blocks"):
        return transformer.blocks
    if hasattr(transformer, "block_groups"):
        blocks = []
        for group in transformer.block_groups:
            if hasattr(group, "blocks"):
                blocks.extend(list(group.blocks))
            else:
                blocks.append(group)
        return blocks
    raise AttributeError("Cannot locate Molmo transformer blocks.")


def _ensure_molmo_config_aliases(config):
    if config is None:
        return None
    alias_pairs = [
        ("hidden_size", ["d_model"]),
        ("intermediate_size", ["mlp_hidden_size"]),
        ("num_attention_heads", ["n_heads"]),
        ("num_key_value_heads", ["effective_n_kv_heads", "n_kv_heads"]),
        ("num_hidden_layers", ["n_layers"]),
        ("max_position_embeddings", ["max_sequence_length"]),
    ]
    for target, sources in alias_pairs:
        if not hasattr(config, target) or getattr(config, target) is None:
            for source in sources:
                if hasattr(config, source) and getattr(config, source) is not None:
                    try:
                        setattr(config, target, getattr(config, source))
                    except Exception:
                        pass
                    break
    if not hasattr(config, "num_key_value_heads") or getattr(config, "num_key_value_heads") is None:
        if hasattr(config, "num_attention_heads"):
            try:
                setattr(config, "num_key_value_heads", getattr(config, "num_attention_heads"))
            except Exception:
                pass
    return config


def _molmo_lm_config(model):
    outer_config = _ensure_molmo_config_aliases(getattr(model, "config", None))
    try:
        inner_config = _ensure_molmo_config_aliases(getattr(_molmo_inner_model(model), "config", None))
    except AttributeError:
        inner_config = None
    if outer_config is not None and hasattr(outer_config, "hidden_size"):
        return outer_config
    if inner_config is not None:
        return inner_config
    raise AttributeError("Cannot locate Molmo language config.")


def _clone_linear_rows(linear, start, end):
    cloned = nn.Linear(
        linear.in_features,
        end - start,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    cloned.weight.data.copy_(linear.weight.data[start:end].detach())
    if linear.bias is not None:
        cloned.bias.data.copy_(linear.bias.data[start:end].detach())
    return cloned


class _MolmoSplitQKV(nn.Module):
    def __init__(self, block, att_proj, attn_out, config):
        super().__init__()
        object.__setattr__(self, "_owner_block", block)
        self.config = config
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(config, "d_model")
        num_heads = getattr(config, "num_attention_heads", None)
        if num_heads is None:
            num_heads = getattr(config, "n_heads")
        hidden_size = int(hidden_size)
        num_heads = int(num_heads)
        head_dim = hidden_size // num_heads
        num_kv_heads = getattr(config, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = getattr(config, "effective_n_kv_heads", getattr(config, "n_kv_heads", num_heads))
        num_kv_heads = int(num_kv_heads)
        fused_dims = getattr(block, "fused_dims", None)
        if fused_dims is None:
            fused_dims = (hidden_size, num_kv_heads * head_dim, num_kv_heads * head_dim)
        self.fused_dims = tuple(int(x) for x in fused_dims)
        q_end = self.fused_dims[0]
        k_end = q_end + self.fused_dims[1]
        v_end = k_end + self.fused_dims[2]
        self.q_proj = _clone_linear_rows(att_proj, 0, q_end)
        self.k_proj = _clone_linear_rows(att_proj, q_end, k_end)
        self.v_proj = _clone_linear_rows(att_proj, k_end, v_end)
        self.o_proj = attn_out

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "o_proj":
            block = self.__dict__.get("_owner_block")
            if block is not None:
                block.__dict__["attn_out"] = value

    def forward(self, x):
        return torch.cat(
            (self.q_proj(x), self.k_proj(x), self.v_proj(x)),
            dim=-1,
        )


class _MolmoMLPAdapter(nn.Module):
    def __init__(self, block, ff_proj, ff_out, config):
        super().__init__()
        object.__setattr__(self, "_owner_block", block)
        self.config = config
        out_features = ff_proj.out_features
        if out_features % 2 != 0:
            raise ValueError(f"Molmo ff_proj output must be even for SwiGLU split, got {out_features}.")
        intermediate_size = out_features // 2
        self.up_proj = _clone_linear_rows(ff_proj, 0, intermediate_size)
        self.gate_proj = _clone_linear_rows(ff_proj, intermediate_size, out_features)
        self.down_proj = ff_out

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "down_proj":
            block = self.__dict__.get("_owner_block")
            if block is not None:
                block.__dict__["ff_out"] = value

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _molmo_checkpoint(block, fn, *args, **kwargs):
    checkpoint_fn = getattr(block, "_activation_checkpoint_fn", None)
    if checkpoint_fn is not None:
        return checkpoint_fn(fn, *args, **kwargs)
    return fn(*args, **kwargs)


def _molmo_dropout(block, x, drop_mask=None):
    dropout = getattr(block, "dropout", None)
    if dropout is None:
        return x
    if drop_mask is not None:
        try:
            return dropout(x, drop_mask=drop_mask)
        except TypeError:
            try:
                return dropout(x, drop_mask)
            except TypeError:
                pass
    return dropout(x)


def _molmo_block_forward(
    self,
    x,
    attention_bias=None,
    position_ids=None,
    layer_past=None,
    use_cache=False,
    drop_mask=None,
    **kwargs,
):
    if attention_bias is None:
        attention_bias = kwargs.get("attention_mask", None)
    if layer_past is None:
        layer_past = kwargs.get("layers_past", None)
    norm_after = bool(getattr(self.config, "norm_after", False))

    if not norm_after:
        atten_in = _molmo_checkpoint(self, self.attn_norm, x)
    else:
        atten_in = x
    qkv = self.self_attn(atten_in)

    clip_qkv = getattr(self.config, "clip_qkv", None)
    if clip_qkv is not None:
        qkv = qkv.clamp(min=-clip_qkv, max=clip_qkv)
    q, k, v = qkv.split(self.self_attn.fused_dims, dim=-1)

    att, cache = _molmo_checkpoint(
        self,
        self.attention,
        q,
        k,
        v,
        attention_bias,
        position_ids=position_ids,
        layer_past=layer_past,
        use_cache=use_cache,
    )

    if norm_after:
        att = _molmo_checkpoint(self, self.attn_norm, att)
    x = x + _molmo_dropout(self, att, drop_mask=drop_mask)

    residual = x
    if not norm_after:
        x = _molmo_checkpoint(self, self.ff_norm, x)
    x = self.mlp(x)
    if norm_after:
        x = _molmo_checkpoint(self, self.ff_norm, x)
    x = residual + _molmo_dropout(self, x, drop_mask=drop_mask)
    return x, cache


def _prepare_molmo_block(block, config):
    if getattr(block, "_wsdv_molmo_compat", False):
        return
    _ensure_molmo_config_aliases(config)
    _ensure_molmo_config_aliases(getattr(block, "config", None))

    if not hasattr(block, "self_attn") and hasattr(block, "att_proj") and hasattr(block, "attn_out"):
        att_proj = block.att_proj
        attn_out = block.attn_out
        self_attn = _MolmoSplitQKV(block, att_proj, attn_out, getattr(block, "config", config))
        for name in ["att_proj", "attn_out"]:
            block._modules.pop(name, None)
            block.__dict__.pop(name, None)
        block.self_attn = self_attn
        block.__dict__["att_proj"] = self_attn
        block.__dict__["attn_out"] = self_attn.o_proj

    if not hasattr(block, "mlp") and hasattr(block, "ff_proj") and hasattr(block, "ff_out"):
        ff_proj = block.ff_proj
        ff_out = block.ff_out
        mlp = _MolmoMLPAdapter(block, ff_proj, ff_out, getattr(block, "config", config))
        for name in ["ff_proj", "ff_out"]:
            block._modules.pop(name, None)
            block.__dict__.pop(name, None)
        block.mlp = mlp
        block.__dict__["ff_proj"] = mlp
        block.__dict__["ff_out"] = mlp.down_proj

    if hasattr(block, "attn_norm"):
        block.__dict__["input_layernorm"] = block.attn_norm
    if hasattr(block, "ff_norm"):
        block.__dict__["post_attention_layernorm"] = block.ff_norm

    if hasattr(block, "self_attn") and hasattr(block, "mlp") and hasattr(block.self_attn, "fused_dims"):
        block.forward = types.MethodType(_molmo_block_forward, block)
        block._wsdv_molmo_compat = True


def prepare_molmo_lm_compat(model):
    if not is_molmo_model(model):
        return model
    if getattr(model, "_wsdv_molmo_compat_prepared", False):
        return model
    config = _molmo_lm_config(model)
    _ensure_molmo_config_aliases(config)
    for block in _molmo_blocks(model):
        _prepare_molmo_block(block, config)
    model._wsdv_molmo_compat_prepared = True
    return model


def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, OPT_MODEL):
        return OPT_MODEL
    elif isinstance(model, LLAVA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, LLAVA_NEXT_HF):
        return LLAVA_NEXT_HF
    elif isinstance(model, LLAMAV_MODLE):
        return LLAMAV_MODLE
    elif isinstance(model, SMOVLM_MODEL):
        return SMOVLM_MODEL
    elif isinstance(model, QWEN25_MODEL):
        return QWEN25_MODEL
    elif isinstance(model, QWEN2_MODEL):
        return QWEN2_MODEL
    elif _is_qwen3_model(model):
        return QWEN3_MODEL_TYPE
    elif isinstance(model, INTERNVL_MODEL):
        return INTERNVL_MODEL
    elif model.__class__.__name__ == "InternVLChatModel":
        return INTERNVL_MODEL
    elif model.__class__.__name__ == 'QWenLMHeadModel':
        return QWEN_VL_MODEL
    elif is_molmo_model(model):
        return MOLMO_MODEL
    elif isinstance(model, BLIP2_INS):
        return BLIP2_INS
    else:
        raise ValueError(f'Unknown model type {model}')

def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass

def get_rope_function_name(model):
    if isinstance(model, LLAMA_MODEL):
        return "apply_rotary_pos_emb"
    if isinstance(model, SMOVLM_MODEL):
        return "apply_rotary_pos_emb"
    if isinstance(model, LLAVA_NEXT_HF):
        return "apply_rotary_pos_emb"
    if isinstance(model, QWEN25_MODEL):
        return "apply_multimodal_rotary_pos_emb"
    if isinstance(model, QWEN2_MODEL):
        return "apply_multimodal_rotary_pos_emb"
    if _is_qwen3_model(model):
        return "apply_rotary_pos_emb"
    if isinstance(model, INTERNVL_MODEL):
        return "apply_rotary_pos_emb" # not sure if this is right
    if model.__class__.__name__ == "InternVLChatModel":
        return "apply_rotary_pos_emb"
    if model.__class__.__name__ == "QWenLMHeadModel":
        return "apply_rotary_pos_emb"
    if is_molmo_model(model):
        return "apply_rotary_pos_emb"
    if isinstance(model, BLIP2_INS):
        return "apply_rotary_pos_emb"
    raise NotImplementedError


def get_layers(model):
    if is_molmo_model(model):
        prepare_molmo_lm_compat(model)
        return _molmo_blocks(model)
    if type(model) == OPT_MODEL:
        return model.model.decoder.layers
    if type(model) == LLAMA_MODEL:
        return model.model.layers
    if type(model) == LLAVA_MODEL:
        return model.model.layers
    if type(model) ==LLAVA_NEXT_HF:
        return model.language_model.layers ## ??? used to be model.language_model.layers ??
    if type(model) == SMOVLM_MODEL:
        return model.model.text_model.layers
    if type(model) == QWEN25_MODEL:
        return model.language_model.layers
    if type(model) == QWEN2_MODEL:
        if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
            return model.language_model.layers
        if (
            hasattr(model, "model")
            and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "layers")
        ):
            return model.model.language_model.layers
        raise AttributeError("Cannot locate Qwen2-VL language_model.layers.")
    if _is_qwen3_model(model):
        if (
            hasattr(model, "model")
            and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "layers")
        ):
            return model.model.language_model.layers
        if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
            return model.language_model.layers
        raise AttributeError("Cannot locate Qwen3-VL language_model.layers.")
    if type(model) == INTERNVL_MODEL:
        return model.language_model.layers
    if model.__class__.__name__ == "InternVLChatModel":
        return model.language_model.model.layers
    if model.__class__.__name__ == "QWenLMHeadModel":
        return model.transformer.h
    if type(model) == BLIP2_INS:
        return model.language_model.model.layers
    
    raise NotImplementedError

def get_llava(model_name, hf_token=None):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    if 'llava-hf' in model_name.lower():
        import transformers
        from transformers import (
            LlavaNextProcessor,
            LlavaNextForConditionalGeneration,
            AutoProcessor,
            LlavaForConditionalGeneration,
        )
        model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True,
                    device_map='cpu',
                )
        model.seqlen=2048
        processor = LlavaNextProcessor.from_pretrained(model_name)
        return model, 'hf_v16', processor
        
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    tokenizer, model, image_processor, seqlen = load_pretrained_model(
        model_path=model_name,
        model_base=None,
        model_name=get_model_name_from_path(model_name),
        device="cpu",
        # attn_implementation='sdpa',# to ensure catcher catch attention_mask, Look into this
        # **{"use_cache": False},
    )
    model.seqlen = seqlen
    return model, tokenizer, image_processor
    

def get_llama(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_r1(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='bfloat16',
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048 # only affect ppl etc
    return model


def get_opt(model_name):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.OPTForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                        low_cpu_mem_usage=True)
    model.seqlen = model.config.max_position_embeddings
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_model(
    model_name, hf_token=None
):
    if 'llama' in model_name and 'vision' not in model_name:
        return get_llama(model_name, hf_token)
    elif 'blip2-opt' in model_name.lower():
        return get_blip2_opt(model_name, hf_token)
    elif 'opt' in model_name:
        return get_opt(model_name)
    elif 'llava' in model_name:
        return get_llava(model_name, hf_token)
    elif 'llama' in model_name and 'vision'  in model_name:
        return get_llamav(model_name, hf_token)
    elif 'SmolVLM' in model_name:
        return get_smovlm(model_name, hf_token)
    elif 'qwen2.5' in model_name.lower():
        return get_qwen25(model_name, hf_token)
    elif 'qwen3' in model_name.lower():
        return get_qwen3(model_name, hf_token)
    elif 'qwen2' in model_name.lower():
        return get_qwen2(model_name, hf_token)
    elif 'qwen-vl' in model_name.lower():
        return get_qwen(model_name, hf_token)
    elif 'internvl' in model_name.lower():
        return get_internvl(model_name, hf_token)
    elif 'instructblip' in model_name.lower():
        return get_blip2_ins(model_name, hf_token)
    elif 'molmo' in model_name.lower():
        return get_molmo(model_name, hf_token)
    elif 'llama' in model_name.lower() and 'r1' in model_name.lower():
        return get_r1(model_name, hf_token)
    elif 'r1' in model_name.lower():
        return get_r1(model_name, hf_token)
    else:
        raise ValueError(f'Unknown model {model_name.lower()}')

def get_molmo(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM, AutoProcessor
    load_kwargs = {"token": hf_token} if hf_token is not None else {}
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        **load_kwargs,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='cpu',
        **load_kwargs,
    )
    model.seqlen = 2048
    prepare_molmo_lm_compat(model)
    
    return model, "molmo", processor


def get_internvl(model_name, hf_token):
    from transformers import AutoTokenizer, AutoModel
    model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map='cpu',
            trust_remote_code=True)
    model.seqlen = 2048
    processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    return model, processor


def get_blip2_opt(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map='cpu')
    model.seqlen = 2048
    processor = Blip2Processor.from_pretrained(model_name)
    return model, processor

def get_blip2_ins(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map='cpu')
    model.seqlen = 2048
    processor = InstructBlipProcessor.from_pretrained(model_name)
    return model, processor

def get_qwen(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import PreTrainedModel

    # Completely disable HF's weight initialization pass, which is what
    # triggers the Resampler/_initialize_weights mess for qwen-vl-chat
    def safe_initialize_weights(self, *args, **kwargs):
        # no-op: assume checkpoint already provides needed weights
        return

    PreTrainedModel.initialize_weights = safe_initialize_weights
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map='cpu',
        trust_remote_code=True).eval()
    model.seqlen = 2048
    processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, "qwenvl", processor

def get_qwen25(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map='cpu',
    )
    model.seqlen = 2048
    processor = AutoProcessor.from_pretrained(model_name)
    return model, "qwen2vl", processor

def get_qwen2(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map='cpu',
    )
    model.seqlen = 2048
    processor = Qwen2VLProcessor.from_pretrained(model_name)
    return model, "qwen2vl", processor

def get_qwen3(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map='cpu',
        trust_remote_code=True,
    )
    model.seqlen = 32768
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    return model, "qwen3vl", processor

def get_smovlm(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoProcessor, Idefics3ForConditionalGeneration
    import os.path as osp
    def splitlen(s, sym='/'):
        return len(s.split(sym))
    assert osp.exists(model_name) or splitlen(model_name) == 2
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='cpu',
    )
    model.seqlen = 2048
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def get_llamav(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import MllamaForConditionalGeneration, MllamaProcessor
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map='cpu',
    )
    processor = MllamaProcessor.from_pretrained(model_name, use_safetensors=True)
    return model, processor


def get_model_type(model):
    if isinstance(model, OPT_MODEL):
        model_type = OPT_MODEL
    elif isinstance(model, LLAMA_MODEL):
        model_type = LLAMA_MODEL
    elif isinstance(model, LLAVA_MODEL):
        model_type = LLAMA_MODEL # for Llava LM usage only, without changing code much
    elif isinstance(model, LLAVA_NEXT_HF):
        model_type = LLAVA_NEXT_HF
    elif isinstance(model, LLAMAV_MODLE):
        model_type = LLAMAV_MODLE
    elif isinstance(model, SMOVLM_MODEL):
        return SMOVLM_MODEL
    elif isinstance(model, QWEN25_MODEL):
        model_type = QWEN25_MODEL
    elif isinstance(model, QWEN2_MODEL):
        model_type = QWEN2_MODEL
    elif _is_qwen3_model(model):
        model_type = QWEN3_MODEL_TYPE
    elif isinstance(model, INTERNVL_MODEL):
        model_type = INTERNVL_MODEL
    elif model.__class__.__name__ == "InternVLChatModel":
        model_type = INTERNVL_MODEL
    elif model.__class__.__name__ == "QWenLMHeadModel":
        model_type = QWEN_VL_MODEL
    elif is_molmo_model(model):
        model_type = MOLMO_MODEL
    elif isinstance(model, BLIP2_INS):
        model_type = BLIP2_INS
    else:
        raise ValueError(f'Unknown model type {model}')
    return model_type


def get_model_typevit(model):
    if isinstance(model, OPT_MODEL):
        model_type = OPT_MODEL
    elif type(model) == LLAVA_MODEL:
        model_type = LLAVA_MODEL
    elif type(model) == LLAVA_NEXT_HF:
        model_type = LLAVA_MODEL # to have vit different from llava? # will this cause projector problem?
    elif type(model) == LLAMAV_MODLE:
        model_type = LLAMAV_MODLE
    elif type(model) == SMOVLM_MODEL:
        model_type =  SMOVLM_MODEL
    elif type(model) == QWEN25_MODEL:
        model_type = QWEN25_MODEL
    elif type(model) == QWEN2_MODEL:
        model_type = QWEN2_MODEL
    elif _is_qwen3_model(model):
        model_type = QWEN3_MODEL_TYPE
    elif type(model) == INTERNVL_MODEL:
        model_type = INTERNVL_MODEL
    elif model.__class__.__name__ == "InternVLChatModel":
        model_type = INTERNVL_MODEL
    elif model.__class__.__name__ == "QWenLMHeadModel":
        model_type = QWEN_VL_MODEL
    elif is_molmo_model(model):
        model_type = MOLMO_MODEL
    elif isinstance(model, BLIP2_INS):
        model_type = BLIP2_INS
    else:
        raise ValueError(f'Unknown model type {model}')
    return model_type

def get_embeddings(model, model_type) -> list[torch.nn.Module]:
    if model_type == LLAMA_MODEL:
        return [model.model.embed_tokens]
    elif model_type == MOLMO_MODEL:
        prepare_molmo_lm_compat(model)
        return [_molmo_transformer(model).wte]
    elif model_type == OPT_MODEL:
        return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    elif model_type == SMOVLM_MODEL:
        return [model.model.text_model.embed_tokens]
    elif model_type == LLAVA_NEXT_HF:
        return [model.language_model.embed_tokens] ### used to be model.language_model.embed_tokens ???
    elif model_type == QWEN25_MODEL:
        return [model.language_model.embed_tokens]
    elif model_type == QWEN2_MODEL:
        if hasattr(model, "language_model") and hasattr(model.language_model, "embed_tokens"):
            return [model.language_model.embed_tokens]
        if (
            hasattr(model, "model")
            and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "embed_tokens")
        ):
            return [model.model.language_model.embed_tokens]
        raise AttributeError("Cannot locate Qwen2-VL embed_tokens.")
    elif model_type == QWEN3_MODEL_TYPE:
        if (
            hasattr(model, "model")
            and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "embed_tokens")
        ):
            return [model.model.language_model.embed_tokens]
        if hasattr(model, "language_model") and hasattr(model.language_model, "embed_tokens"):
            return [model.language_model.embed_tokens]
        raise AttributeError("Cannot locate Qwen3-VL embed_tokens.")
    elif model_type == INTERNVL_MODEL:
        return [model.language_model.embed_tokens]
    elif model.__class__.__name__ == "QWenLMHeadModel":
        return [model.transformer.attn_pool]
    elif model_type == BLIP2_INS:
        return [model.language_model.model.embed_tokens]
    else:
        raise ValueError(f'Unknown model type {model_type}')

def get_projector(model, model_type) -> list[torch.nn.Module]:
    if model_type == LLAVA_MODEL or model_type == LLAMA_MODEL:
        return [model.model.mm_projector]
    elif model_type == LLAMAV_MODLE:
        return [model.multi_modal_projector]
    elif model_type == LLAVA_NEXT_HF:
        return [model.multi_modal_projector]
    elif model_type == SMOVLM_MODEL:
        return [model.model.connector.modality_projection.proj] # no bias
    elif model_type == QWEN25_MODEL:
        return [model.model.visual.merger.mlp]
    elif model_type == QWEN2_MODEL:
        return [model.model.visual.merger.mlp]
    elif model_type == QWEN3_MODEL_TYPE:
        return [model.model.visual.merger.mlp]
    elif model_type == INTERNVL_MODEL:
        return [model.mlp1]
    elif model.__class__.__name__ == "QWenLMHeadModel":
        return [model.transformer.wte]
    elif model_type == MOLMO_MODEL:
        inner = _molmo_inner_model(model)
        if hasattr(inner, "vision_backbone") and hasattr(inner.vision_backbone, "image_projector"):
            return [inner.vision_backbone.image_projector]
        return []
    elif model_type == BLIP2_INS:
        return None
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_transformer_layers(model, model_type):
    if model_type == LLAMA_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == MOLMO_MODEL:
        prepare_molmo_lm_compat(model)
        return [layer for layer in _molmo_blocks(model)]
    elif model_type == OPT_MODEL:
        return [layer for layer in model.model.decoder.layers]
    elif model_type == LLAVA_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == SMOVLM_MODEL:
        return [layer for layer in model.model.text_model.layers]
    elif model_type == LLAVA_NEXT_HF:
        return [layer for layer in model.language_model.layers] ## used to be model.language_model.layers ???
    elif model_type == QWEN25_MODEL:
        return [layer for layer in model.model.language_model.layers]
    elif model_type == QWEN2_MODEL:
        return [layer for layer in model.model.language_model.layers]
    elif model_type == QWEN3_MODEL_TYPE:
        return [layer for layer in model.model.language_model.layers]
    elif model_type == INTERNVL_MODEL:
        return [layer for layer in model.language_model.model.layers]
    elif model.__class__.__name__ == "QWenLMHeadModel":
        return [layer for layer in model.transformer.h]
    elif model_type == BLIP2_INS:
        return [layer for layer in model.language_model.model.layers]
    else:
        raise ValueError(f'Unknown model type {model_type}')

def get_vit_layers(model, model_type):
    if type(model) == LLAVA_NEXT_HF: # to surpass vit model type in llava_model for attn/mlp rotation simplicity
        return [layer for layer in model.vision_tower.vision_model.encoder.layers] # do not know if this works
    elif model_type == LLAVA_MODEL:
        return [layer for layer in model.model.vision_tower.vision_tower.vision_model.encoder.layers]
    elif model_type == LLAMAV_MODLE:
        return [layer for layer in model.vision_model.transformer.layers] + [layer for layer in model.vision_model.global_transformer.layers]
    elif model_type == SMOVLM_MODEL:
        return [layer for layer in model.model.vision_model.encoder.layers]
    elif model_type == QWEN25_MODEL:
        return [layer for layer in model.model.visual.blocks]
    elif model_type == QWEN2_MODEL:
        return [layer for layer in model.model.visual.blocks]
    elif model_type == QWEN3_MODEL_TYPE:
        return [layer for layer in model.model.visual.blocks]
    elif model_type == INTERNVL_MODEL:
        return [layer for layer in model.vision_model.encoder.layers]
    elif model.__class__.__name__ == "QWenLMHeadModel":
        return [layer for layer in model.transformer.visual.transformer.resblocks]
    elif model_type == MOLMO_MODEL:
        inner = _molmo_inner_model(model)
        if hasattr(inner, "vision_backbone") and hasattr(inner.vision_backbone, "image_vit"):
            vit = inner.vision_backbone.image_vit
            if hasattr(vit, "transformer") and hasattr(vit.transformer, "resblocks"):
                return [layer for layer in vit.transformer.resblocks]
            if hasattr(vit, "blocks"):
                return [layer for layer in vit.blocks]
        raise ValueError(f'Unknown Molmo vit layer layout {type(model)}')
    elif model_type == BLIP2_INS:
        return [layer for layer in model.vision_model.encoder.layers]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_vit_config(model):
    if type(model) == LLAVA_MODEL:
        return model.model.vision_tower.config
    elif type(model) == LLAVA_NEXT_HF:
        return model.vision_tower.config
    elif type(model) == QWEN25_MODEL:
        return model.model.config.vision_config
    elif type(model) == QWEN2_MODEL:
        return model.model.config.vision_config
    elif _is_qwen3_model(model):
        return model.model.config.vision_config
    elif type(model) == INTERNVL_MODEL:
        return model.config.vision_config
    elif model.__class__.__name__ == "QWenLMHeadModel":
        return model.config.vision_config
    elif type(model) == BLIP2_INS:
        return model.config.vision_config
    else:
        raise ValueError(f'Unknown model type {type(model)}')

def get_lm_config(model):
    if is_molmo_model(model):
        prepare_molmo_lm_compat(model)
        return _molmo_lm_config(model)
    if type(model) == LLAVA_NEXT_HF:
        return model.config.text_config
    elif type(model) == BLIP2_INS:
        return model.config.text_config
    else:
        return model.config
    

def get_lm_head(model, model_type):
    if model_type in [LLAMA_MODEL, SMOVLM_MODEL]:
        return model.lm_head
    elif model_type == MOLMO_MODEL:
        transformer = _molmo_transformer(model)
        if hasattr(transformer, "ff_out"):
            return transformer.ff_out
        if hasattr(model, "get_output_embeddings"):
            return model.get_output_embeddings()
        raise AttributeError("Cannot locate Molmo lm_head.")
    elif model_type == OPT_MODEL:
        return model.lm_head
    elif model_type == LLAVA_NEXT_HF:
        # HF LlavaNext variants may expose lm_head on the outer model
        # or under language_model depending on transformers version.
        if hasattr(model, "lm_head"):
            return model.lm_head
        if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
            return model.language_model.lm_head
        raise AttributeError("Cannot locate lm_head for LLAVA_NEXT_HF model.")
    elif model_type == QWEN25_MODEL:
        return model.lm_head
    elif model_type == QWEN2_MODEL:
        if hasattr(model, "lm_head"):
            return model.lm_head
        if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
            return model.language_model.lm_head
        if (
            hasattr(model, "model")
            and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "lm_head")
        ):
            return model.model.language_model.lm_head
        raise AttributeError("Cannot locate Qwen2-VL lm_head.")
    elif model_type == QWEN3_MODEL_TYPE:
        if hasattr(model, "lm_head"):
            return model.lm_head
        if (
            hasattr(model, "model")
            and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "lm_head")
        ):
            return model.model.language_model.lm_head
        if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
            return model.language_model.lm_head
        raise AttributeError("Cannot locate Qwen3-VL lm_head.")
    elif model_type == INTERNVL_MODEL:
        return model.language_model.lm_head
    elif model.__class__.__name__ == "QWenLMHeadModel":
        return model.lm_head
    elif model_type == BLIP2_INS:
        return model.language_model.lm_head
    else:
        raise ValueError(f'Unknown model type {model_type}')

def get_pre_head_layernorm(model, model_type):
    if model_type in [LLAMA_MODEL]:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          transformers.models.llama.modeling_llama.LlamaRMSNorm)
    elif model_type == MOLMO_MODEL:
        pre_head_layernorm = _molmo_transformer(model).ln_f
    elif model_type in [SMOVLM_MODEL]:
        pre_head_layernorm = model.model.text_model.norm
        assert isinstance(pre_head_layernorm,
                          transformers.models.llama.modeling_llama.LlamaRMSNorm)
    elif model_type == OPT_MODEL:
        pre_head_layernorm = model.model.decoder.final_layer_norm
        assert pre_head_layernorm is not None
    elif model_type == LLAVA_NEXT_HF:
        pre_head_layernorm = model.language_model.norm  ## used to be model.language_model.norm ???
        assert isinstance(pre_head_layernorm,
                          transformers.models.llama.modeling_llama.LlamaRMSNorm)
    elif model_type == BLIP2_INS:
        pre_head_layernorm = model.language_model.model.norm  ## used to be model.language_model.norm ???
        assert isinstance(pre_head_layernorm,
                          transformers.models.llama.modeling_llama.LlamaRMSNorm)
    elif model_type == QWEN25_MODEL:
        pre_head_layernorm = model.model.language_model.norm
        assert isinstance(pre_head_layernorm,
                          transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm)
    elif model_type == QWEN2_MODEL:
        pre_head_layernorm = model.model.language_model.norm
        assert isinstance(pre_head_layernorm,
                          transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm)
    elif model_type == QWEN3_MODEL_TYPE:
        pre_head_layernorm = model.model.language_model.norm
    elif model_type == INTERNVL_MODEL:
        pre_head_layernorm = model.language_model.norm
        assert isinstance(pre_head_layernorm,
                          transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm)
    elif model.__class__.__name__ == "QWenLMHeadModel":
        pre_head_layernorm = model.transformer.ln_f
        print('do not know where to validate the isinstance')
    else:
        raise ValueError(f'Unknown model type {model_type}')
    return pre_head_layernorm

def get_mlp_bottleneck_size(model):
    model_type = get_model_type(model)
    if model_type == LLAMA_MODEL:
        return model.config.intermediate_size
    elif model_type == OPT_MODEL:
        return model.config.ffn_dim
    else:
        raise ValueError(f'Unknown model type {model_type}')

def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


class RMSLN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


class LN_(torch.nn.Module):
    def __init__(self, ori_layer_norm) -> None:
        super().__init__()
        self.ln = ori_layer_norm


    def forward(self, x):
        out = self.ln(x)
        outdtype = out.dtype
        out_ = out.double()
        out_ = out_ - out_.mean(dim=-1, keepdim=True)
        return out_.to(outdtype)


class LNRotWrapper(torch.nn.Module):
    def __init__(self, module:None):
        super().__init__()
        assert isinstance(module, LN_) or isinstance(module, torch.nn.LayerNorm)
        self.module = module
        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.online_random_had = False
        self.had_dim = 0
        self.fp32_had = False
    
    def forward(self, x):
        x_dtype = x.dtype
        x = self.module(x)
        if self.online_random_had:
            if self.fp32_had:
                x = (x.double()@self.had_K).to(x_dtype)
            else:
                x = x@self.had_K.to(x_dtype)
        # Rotate, if needed
        if self.online_full_had:
            from . import hadamard_utils
            if self.fp32_had: # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else: # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)
        return x.to(x_dtype)

def get_layer_io_save_path(args):
    return os.path.join(args.save_path, 'layer_io', f'{args.layer_idx:03d}.pt')

def capture_layer_io(model_type, layer, layer_input):
    def hook_factory(module_name, captured_vals, is_input):
        def hook(module, input, output):
            if is_input:
                captured_vals[module_name].append(input[0].detach().cpu())
            else:
                captured_vals[module_name].append(output.detach().cpu())
        return hook

    handles = []

    if model_type == LLAMA_MODEL:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'o_proj': [],
            'gate_proj': [],  # up_proj has the same input as gate_proj
            'down_proj': []
        }

        captured_outputs = {
            'v_proj': [],
        }

        for name in captured_inputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))

    elif model_type == OPT_MODEL:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'out_proj': [],
            'fc1': [],
            'fc2': []
        }
        captured_outputs = {
            'v_proj': [],
        }
        for name in captured_inputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))
    else:
        raise ValueError(f'Unknown model type {model_type}')

    # Process each sequence in the batch one by one to avoid OOM.
    for seq_idx in range(layer_input.shape[0]):
        # Extract the current sequence across all dimensions.
        seq = layer_input[seq_idx:seq_idx + 1].to(utils.DEV)
        # Perform a forward pass for the current sequence.
        layer(seq)

    # After processing all sequences, concatenate the accumulated inputs for each sub-layer across the batch.
    for module_name in captured_inputs:
        captured_inputs[module_name] = torch.cat(captured_inputs[module_name], dim=0)
    for module_name in captured_outputs:
        captured_outputs[module_name] = torch.cat(captured_outputs[module_name], dim=0)

    # Cleanup.
    for h in handles:
        h.remove()

    return {
        'input': captured_inputs,
        'output': captured_outputs
    }


class Layerwrapper(torch.nn.Module):
    def __init__(self, module:None, R1=None, R2=None, R3=None, R4=None):
        super().__init__()
        self.module = module
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.R4 = R4

    def init_quant(self, args):
        # act_quantizer
        # quant_utils.add_actquant(model)
        # weight quantizer
        if self.R3 is not None:
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                args.w_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
            )
            self.module.mlp.fc1.quantizer = quantizer
        if self.R4 is not None:
            # for now just add R4 actquantizer
            quantizer = quant_utils.ActQuantizer()
            quantizer.configure(
                bits=args.a_bits,
                groupsize=args.a_groupsize, # the group size should further refer to main
                sym=not(args.a_asym),
                clip_ratio=args.a_clip_ratio)
            self.module.mlp.fc2.actquantizer = quantizer
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                args.w_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
            )
            self.module.mlp.fc2.quantizer = quantizer

    def init_rot(self):
        # here add linear layers
        if self.R1 is not None:
            pass
            # x = x @ self.R1
            # todo fuse mlp output
        if self.R2 is not None:
            # todo fuse qkv, fc1 input
            pass
        if self.R3 is not None:
            self.module.mlp.fc1 = QuantizeLinear(self.module.mlp.fc1, Routput=self.R3)
        if self.R4 is not None:
            self.module.mlp.fc2 = QuantizeLinear(self.module.mlp.fc2, Rinput=self.R4)

    def reset_rot(self):
        if self.R3 is not None:
            self.module.mlp.fc1 = self.module.mlp.fc1.module
        if self.R4 is not None:
            self.module.mlp.fc2 = self.module.mlp.fc2.module
            
    def forward(self, x, *args, **kwargs):

        return self.module(x, *args, **kwargs)
    

class mlpLayerwrapper(torch.nn.Module):
    def __init__(self, module:None, R1=None, R2=None, R3=None, R4=None, sq=None):
        super().__init__()
        self.module = module
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.R4 = R4
        self.sq = sq
        if self.sq is not None:
            self.sq = sq[0]
            self.sq1 = sq[1]
        else:
            self.sq1 = None

    def init_quant(self, args):
        # act_quantizer
        # quant_utils.add_actquant(model)
        # weight quantizer
        if self.R3 is not None:
            pass
        if self.R4 is not None:
            # for now just add R4 actquantizer
            quantizer = quant_utils.ActQuantizer()
            quantizer.configure(
                bits=args.a_bits,
                groupsize=args.a_groupsize, # the group size should further refer to main
                sym=not(args.a_asym),
                clip_ratio=args.a_clip_ratio)
            self.module.fc2.actquantizer = quantizer
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                args.w_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
            )
            self.module.fc2.quantizer = quantizer

    def init_rot(self):
        # here add linear layers
        if self.R1 is not None:
            pass
            # x = x @ self.R1
            # todo fuse mlp output
        if self.R2 is not None:
            # todo fuse qkv, fc1 input
            pass
        if self.R3 is not None:
            self.module.fc1 = QuantizeLinear(self.module.fc1, Routput=self.R3, sq=self.sq1)
        if self.R4 is not None:
            self.module.fc2 = QuantizeLinear(self.module.fc2, Rinput=self.R4, sq=self.sq)

    def init_onlinerot(self, args):
        if self.R4 is not None:
            self.module.fc2 = QuantizeLinear(self.module.fc2, Rinput=self.R4, sq=self.sq)
            if args.rot_dim:
                self.module.fc2.online_partial_had = True
                self.module.fc2.had_dim = args.rot_dim
            else:
                self.module.fc2.online_random_had = True 
            self.module.fc2.fp32_had = args.fp32_had
    def reset_rot(self):
        if self.R3 is not None:
            self.module.fc1 = self.module.fc1.module
        if self.R4 is not None:
            self.module.fc2 = self.module.fc2.module
    def reset_onlinerot(self):
        if self.R4 is not None:
            self.module.fc2 = self.module.fc2.module

    def forward(self, x, *args, **kwargs):

        return self.module(x, *args, **kwargs)


from torch._tensor import Tensor
from . import hadamard_utils
import fast_hadamard_transform
import math

class QuantizeLinear(torch.nn.Module):
    def __init__(self, module, Rinput=None, Routput=None, sq = None):
        super().__init__()
        self.module = module
        self.Rinput = Rinput
        self.Routput = Routput
        self.sq = sq
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.online_random_had = False
        self.had_dim = 0
        self.fp32_had = False

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        x_dtype = input.dtype
        if self.online_random_had:
            if self.fp32_had:
                input = input.double()@self.Rinput.to(x_dtype)
            else:
                input = input @ self.Rinput.to(x_dtype)
        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!
            
            if self.fp32_had:
                input = input.float()
                
            init_shape = input.shape
            input = (input.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim) @ self.Rinput.to(input.dtype))
            if self.fp32_had:
                input = input.to(x_dtype)
            input = input.reshape(init_shape)
        if hasattr(self, "actquantizer"):
            x_dtype = input.dtype
            if self.actquantizer.bits < 16: #Quantize, if needed
                self.actquantizer.find_params(input)
                input = self.actquantizer(input).to(x_dtype)
                self.actquantizer.free()
        weight = self.module.weight
        bias = self.module.bias
        if self.Routput is not None:
            W = self.module
            dtype = W.weight.dtype
            if self.sq is not None:
                weight = (weight.double() / self.sq.double()).to(dtype)
                if W.bias is not None:
                    bias = (bias.double() / self.sq.double()).to(dtype)
            if W.weight.shape[0] == self.Routput.shape[0]:
                weight = (self.Routput.T.to(torch.float64) @ weight.to(torch.float64)).to(dtype)
                if W.bias is not None:
                    bias = (self.Routput.T.to(torch.float64) @ bias.to(torch.float64)).to(dtype)
            else:
                # do not know if this non-transpose thing works or not
                hdim = self.Routput.shape[0]
                init_shape = weight.shape
                temp = weight.reshape(init_shape[0]//hdim, hdim, -1)
                temp = (self.Routput.T.to(torch.float64) @  temp.to(torch.float64)).to(dtype)
                weight = temp.reshape(init_shape)

                if W.bias is not None:
                    init_shape = bias.shape
                    temp = bias.reshape(hdim, init_shape[0]//hdim)
                    temp = (self.Routput.T.to(torch.float64) @  temp.to(torch.float64)).to(dtype)
                    bias = temp.reshape(init_shape)

        if self.Rinput is not None:
            W = self.module
            dtype = W.weight.dtype
            if self.sq is not None:
                weight = weight * self.sq.to(dtype)
            if W.weight.shape[1] == self.Rinput.shape[0]:
                weight = (weight.to(torch.float64) @ self.Rinput.to(torch.float64)).to(dtype)
            else:
                hdim = self.Rinput.shape[0]
                init_shape = weight.shape
                temp = weight.reshape(-1, init_shape[0]//hdim, hdim)
                temp = (temp.to(torch.float64) @ self.Rinput.to(torch.float64)).to(dtype)
                weight = temp.reshape(init_shape)
        if hasattr(self, "quantizer"):
            dtype = weight.dtype
            self.quantizer.find_params(weight.data)
            weight = self.quantizer.quantize(weight).to(dtype)

        return torch.nn.functional.linear(input, weight, bias)
