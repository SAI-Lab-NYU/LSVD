from . import model_utils
import torch
import torch.nn as nn
import typing
from . import utils
import transformers
import tqdm
import math
from . import quant_utils
from . import hadamard_utils
from . import gptq_utils
from .hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2, apply_exact_had_to_linear_nopow2
from fast_hadamard_transform import hadamard_transform
from .optimizer_utils import SGDG
import logging


def _lm_rmsnorm_type(model, model_type):
    if model_type == model_utils.SMOVLM_MODEL:
        return type(model_utils.get_pre_head_layernorm(model, model_type))
    if model_type == model_utils.QWEN2_MODEL:
        return type(_get_pre_head_layernorm(model, model_type))
    if model_type == model_utils.MOLMO_MODEL:
        return type(model_utils.get_pre_head_layernorm(model, model_type))
    return transformers.models.llama.modeling_llama.LlamaRMSNorm


def _is_qwen2vl_model_type(model_type):
    return model_type == model_utils.QWEN2_MODEL


def _is_llama_style_qwen2vl_lm(model_type):
    return model_type in [model_utils.LLAMA_MODEL, model_utils.LLAVA_NEXT_HF, model_utils.MOLMO_MODEL] or _is_qwen2vl_model_type(model_type)


def _qwen2vl_language_model(model):
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
    ):
        return model.model.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    raise AttributeError("Cannot locate Qwen2-VL language_model.")


def _get_lm_config(model, model_type):
    if _is_qwen2vl_model_type(model_type):
        if hasattr(model, "config") and hasattr(model.config, "text_config"):
            return model.config.text_config
        return _qwen2vl_language_model(model).config
    return model_utils.get_lm_config(model)


def _get_transformer_layers(model, model_type):
    if _is_qwen2vl_model_type(model_type):
        return [layer for layer in _qwen2vl_language_model(model).layers]
    return model_utils.get_transformer_layers(model, model_type)


def _get_pre_head_layernorm(model, model_type):
    if _is_qwen2vl_model_type(model_type):
        return _qwen2vl_language_model(model).norm
    return model_utils.get_pre_head_layernorm(model, model_type)


def _lm_rmsn_hidden_size(model, model_type):
    if _is_qwen2vl_model_type(model_type):
        return _get_lm_config(model, model_type).hidden_size
    if model_type in [model_utils.LLAMA_MODEL, model_utils.LLAVA_NEXT_HF, model_utils.MOLMO_MODEL]:
        return _get_lm_config(model, model_type).hidden_size
    return model.model.text_model.config.hidden_size


def _is_molmo_post_norm_model(model, model_type):
    return (
        model_type == model_utils.MOLMO_MODEL
        and bool(getattr(_get_lm_config(model, model_type), "norm_after", False))
    )


class _SharedRotationCache:
    def __init__(self, Q: torch.Tensor):
        self.Q = Q.detach().to(device="cpu", dtype=torch.float32)
        self._cache = {}

    def get(self, x: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        key = (device.type, device.index, dtype, transpose)
        cached = self._cache.get(key)
        if cached is None:
            cached = self.Q.T if transpose else self.Q
            cached = cached.to(device=device, dtype=dtype)
            self._cache[key] = cached
        return cached


class _MolmoPostNormRotationWrapper(nn.Module):
    def __init__(self, module: nn.Module, rotation_cache: _SharedRotationCache):
        super().__init__()
        self.module = module
        self.rotation_cache = rotation_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        Q = self.rotation_cache.get(x, transpose=False)
        QT = self.rotation_cache.get(x, transpose=True)
        y = self.module(torch.matmul(x, QT))
        return torch.matmul(y, Q).to(x_dtype)


def _wrap_molmo_post_norm(module: nn.Module, rotation_cache: _SharedRotationCache) -> nn.Module:
    if isinstance(module, _MolmoPostNormRotationWrapper):
        module.rotation_cache = rotation_cache
        return module
    return _MolmoPostNormRotationWrapper(module, rotation_cache)


def _wrap_molmo_post_norms_for_rotation(model, model_type, Q: torch.Tensor) -> None:
    if not _is_molmo_post_norm_model(model, model_type):
        return

    rotation_cache = _SharedRotationCache(Q)
    transformer = model_utils._molmo_transformer(model)

    if hasattr(transformer, "ln_f"):
        transformer.ln_f = _wrap_molmo_post_norm(transformer.ln_f, rotation_cache)

    for block in model_utils._molmo_blocks(model):
        if hasattr(block, "attn_norm"):
            block.attn_norm = _wrap_molmo_post_norm(block.attn_norm, rotation_cache)
            block.__dict__["input_layernorm"] = block.attn_norm
        if hasattr(block, "ff_norm"):
            block.ff_norm = _wrap_molmo_post_norm(block.ff_norm, rotation_cache)
            block.__dict__["post_attention_layernorm"] = block.ff_norm

    logging.info("wrapped Molmo post-norm layers for rotation")


def _replace_lm_rmsnorms_after_fusion(model, model_type):
    if model_type == model_utils.LLAVA_NEXT_HF:
        model_utils.replace_modules(
            model,
            transformers.models.llama.modeling_llama.LlamaRMSNorm,
            lambda _: model_utils.RMSN(model.config.text_config.hidden_size, eps=1e-5),
            replace_layers=False,
        )
    elif _is_qwen2vl_model_type(model_type):
        model_utils.replace_modules(
            _qwen2vl_language_model(model),
            _lm_rmsnorm_type(model, model_type),
            lambda _: model_utils.RMSN(_lm_rmsn_hidden_size(model, model_type), eps=1e-5),
            replace_layers=False,
        )
    elif model_type == model_utils.MOLMO_MODEL:
        hidden_size = _lm_rmsn_hidden_size(model, model_type)
        transformer = model_utils._molmo_transformer(model)

        if hasattr(transformer, "ln_f"):
            transformer.ln_f = model_utils.RMSN(hidden_size, eps=1e-5)

        for block in model_utils._molmo_blocks(model):
            if hasattr(block, "attn_norm"):
                block.attn_norm = model_utils.RMSN(hidden_size, eps=1e-5)
                block.__dict__["input_layernorm"] = block.attn_norm
            if hasattr(block, "ff_norm"):
                block.ff_norm = model_utils.RMSN(hidden_size, eps=1e-5)
                block.__dict__["post_attention_layernorm"] = block.ff_norm
    else:
        model_utils.replace_modules(
            model,
            _lm_rmsnorm_type(model, model_type) if model_type in [model_utils.LLAMA_MODEL, model_utils.SMOVLM_MODEL, model_utils.LLAVA_NEXT_HF] else torch.nn.LayerNorm,
            lambda _: model_utils.RMSN(model.config.hidden_size, eps=1e-5) if model_type in [model_utils.LLAMA_MODEL, model_utils.LLAVA_NEXT_HF] else model_utils.RMSN(model.model.text_model.config.hidden_size, eps=1e-5),
            replace_layers=False,
        )


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    layernorm_bias = getattr(layernorm, 'bias', None)
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()

        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)
        if layernorm_bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(
                        linear.out_features,
                        dtype=torch.float64,
                        device=linear.weight.device,
                    )
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm_bias.double().to(W_.device)
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)

def fuse_ln_linear_localft(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear], args: None) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    fuse the linear operations in Layernorm into the V matrix
    """
    compute_device = utils.get_dev()
    layernorm_bias = getattr(layernorm, 'bias', None)
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        linear_device = linear.weight.device

        # Calculating new weight and bias
        W_ = linear.weight.data.double().to(compute_device)
        V = linear.svd_info_before_rot['V']
        ###### U, V shape:
            # is_perhead
                # U: n, c, c(r)
                # S: n, C(r)
                # V: n, C_in, c(r)
            # original
                # U: Cout, C(r)
                # S: C(r)
                # V: C_in, C(r)

        linear.weight.data = (W_ * layernorm.weight.double().to(compute_device)).to(linear_dtype).to(linear_device)
        if args.is_per_head_svd:
            if isinstance(V, list):
                dtype = V[0].dtype
                device = V[0].device
                V = [(vv.double().to(compute_device) * layernorm.weight.double()[:, None].to(compute_device)).to(dtype).to(device) for vv in V] #[C_in, c(r) *  C_in, 1]
            else:
                dtype = V.dtype
                device = V.device
                V = (V.double().to(compute_device) * layernorm.weight.double()[:, None].to(compute_device)).to(dtype).to(device) #[n, C_in, c(r) *  C_in, 1]
        else:
            dtype = V.dtype
            device = V.device
            V = (V.double().to(compute_device) * layernorm.weight.double()[:, None].to(compute_device)).to(dtype).to(device) #[C_in, C(r) *  C_in, 1]
        if layernorm_bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(
                        linear.out_features,
                        dtype=torch.float64,
                        device=linear.weight.device,
                    )
                )
            linear.bias.data = linear.bias.data.double().to(compute_device) + torch.matmul(
                W_, layernorm_bias.double().to(compute_device)
            )
            linear.bias.data = linear.bias.data.to(linear_dtype).to(linear_device)
        linear.svd_info_before_rot['V'] = V

            
def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)

def bake_mean_into_linearmm(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and adds the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ + W_.mean(dim=-1, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)


def bake_mean_into_linear_return(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    W_mean = W_.mean(dim=-2, keepdim=True)
    linear.weight.data = W_ - W_mean
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        b_mean = b_.mean()
        linear.bias.data = b_ - b_mean
        linear.bias.data = linear.bias.data.to(linear_dtype)
    return (W_mean, b_mean)

def fuse_layer_norms_noeb(model):
    
    model_type = model_utils.get_model_type(model)
    if _is_molmo_post_norm_model(model, model_type):
        logging.info("skip Molmo post-norm fusion; norms will be wrapped during rotation")
        return
    
    kwargs = {'model': model, 'model_type': model_type}
    
        
    layers = _get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        
        # fuse the input layernorms into the linear layers
        if model_type in [model_utils.LLAMA_MODEL, model_utils.SMOVLM_MODEL, model_utils.LLAVA_NEXT_HF, model_utils.MOLMO_MODEL] or _is_qwen2vl_model_type(model_type):
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == model_utils.OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
            
        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
                    
    
    fuse_ln_linear(_get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])
    
    _replace_lm_rmsnorms_after_fusion(model, model_type)
    logging.info('finish lm fuse norm 1e-5')

def fuse_layer_norms_noeb_localft(model, args):
    model_type = model_utils.get_model_type(model)
    if _is_molmo_post_norm_model(model, model_type):
        logging.info("skip Molmo post-norm fusion; norms will be wrapped during rotation")
        return
    
    kwargs = {'model': model, 'model_type': model_type}
    
        
    layers = _get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        
        # fuse the input layernorms into the linear layers
        if model_type in [model_utils.LLAMA_MODEL, model_utils.SMOVLM_MODEL, model_utils.LLAVA_NEXT_HF, model_utils.MOLMO_MODEL] or _is_qwen2vl_model_type(model_type):
            if args.svd_modules in ['all', 'mlp', 'gaup']:
                fuse_ln_linear_localft(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj], args)
            else:
                fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
            if args.svd_modules in ['all', 'qkv', 'attn']:
                fuse_ln_linear_localft(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj], args)
            else:
                fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == model_utils.OPT_MODEL:
            logging.info('opt model only supported all model svd')
            fuse_ln_linear_localft(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj], args)
            fuse_ln_linear_localft(layer.final_layer_norm, [layer.fc1], args)
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
            
        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
                    
    
    fuse_ln_linear(_get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])
    
    _replace_lm_rmsnorms_after_fusion(model, model_type)
    logging.info('finish lm fuse norm 1e-5') 


def fuse_layer_norms_noeb5(model):
    
    model_type = model_utils.get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
        
    layers = model_utils.get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAMA_MODEL:
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == model_utils.OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
            
        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
                    
    
    fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])
    
    model_utils.replace_modules(
        model,
        transformers.models.llama.modeling_llama.LlamaRMSNorm if model_type == model_utils.LLAMA_MODEL else torch.nn.LayerNorm,
        lambda _: model_utils.RMSN(model.config.hidden_size, eps=1e-5),
        replace_layers=False,
    )
    logging.info('finish lm fuse norm 1e-5')


def fuse_layer_norms_noebsvd(model):
    
    model_type = model_utils.get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
        
    layers = model_utils.get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAMA_MODEL:
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj.BLinear, layer.mlp.gate_proj.BLinear])    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj.BLinear, layer.self_attn.k_proj.BLinear, layer.self_attn.v_proj.BLinear])
        elif model_type == model_utils.OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
            
        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
                    
    
    fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])
    
    model_utils.replace_modules(
        model,
        transformers.models.llama.modeling_llama.LlamaRMSNorm if model_type == model_utils.LLAMA_MODEL else torch.nn.LayerNorm,
        lambda _: model_utils.RMSN(model.config.hidden_size),
        replace_layers=False,
    )
    
def fuse_layer_norms_noebsvdkv(model):
    
    model_type = model_utils.get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
        
    layers = model_utils.get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAMA_MODEL:
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj.BLinear, layer.self_attn.v_proj.BLinear])
        elif model_type == model_utils.OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
            
        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
                    
    
    fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])
    
    model_utils.replace_modules(
        model,
        transformers.models.llama.modeling_llama.LlamaRMSNorm if model_type == model_utils.LLAMA_MODEL else torch.nn.LayerNorm,
        lambda _: model_utils.RMSN(model.config.hidden_size),
        replace_layers=False,
    )
 
def fuse_layer_norms_noebsvdqkv(model, svd_modules=None):
    
    model_type = model_utils.get_model_type(model)
    if _is_molmo_post_norm_model(model, model_type):
        logging.info("skip Molmo post-norm SVD-QKV fusion; norms will be wrapped during rotation")
        return
    
    kwargs = {'model': model, 'model_type': model_type}
    
        
    layers = _get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        
        # fuse the input layernorms into the linear layers
        if model_type in [model_utils.LLAMA_MODEL, model_utils.SMOVLM_MODEL, model_utils.LLAVA_NEXT_HF, model_utils.MOLMO_MODEL] or _is_qwen2vl_model_type(model_type):
            if svd_modules in ['all', 'mlp', 'gaup']:
                fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj.BLinear, layer.mlp.gate_proj.BLinear])
            else:
                fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
            if svd_modules in ['all', 'qkv', 'gaup']:
                fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj.BLinear, layer.self_attn.k_proj.BLinear, layer.self_attn.v_proj.BLinear])
            else:
                fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == model_utils.OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
            
        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
                    
    
    fuse_ln_linear(_get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])
    _replace_lm_rmsnorms_after_fusion(model, model_type)
    logging.info('finish svd qkv fuse norm 1e-5')
 

def fuse_layer_norms(model, svd_modules=None):
    
    model_type = model_utils.get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion # why is this?
    for W in model_utils.get_embeddings(**kwargs):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
    layers = model_utils.get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAMA_MODEL:
            if svd_modules in ['all', 'mlp', 'gaup']:
                fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj.BLinear, layer.mlp.gate_proj.BLinear])
            else:
                fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
            if svd_modules in ['all', 'qkv', 'attn']:
                fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj.BLinear, layer.self_attn.k_proj.BLinear, layer.self_attn.v_proj.BLinear])
            else:
                fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == model_utils.OPT_MODEL: # ignore opt svd
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
            
        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
                    
    
    fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])
    
    model_utils.replace_modules(
        model,
        transformers.models.llama.modeling_llama.LlamaRMSNorm if model_type == model_utils.LLAMA_MODEL else torch.nn.LayerNorm,
        lambda _: model_utils.RMSN(model.config.hidden_size),
        replace_layers=False,
    )
    
def fuse_layer_normsvit(model):
    
    model_type = model_utils.get_model_typevit(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion # why is this?
    # for W in model_utils.get_embeddings(**kwargs):
    #     W_ = W.weight.data.double()
    #     W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
    ########## VIT no embedding replace this to pre-LN
    # Replace Pre-LN in vit with substract mean
    vit_model = model.model.vision_tower.vision_tower.vision_model
    vit_model.pre_layrnorm = model_utils.LN_(vit_model.pre_layrnorm)
    layers = model_utils.get_vit_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for idx, layer in enumerate(layers):
        
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAVA_MODEL:
            fuse_ln_linear(layer.layer_norm1, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.layer_norm2, [layer.mlp.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
        if model_type == model_utils.LLAVA_MODEL:
            # here do not include the second last layers
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.mlp.fc2) # here skip the second last one, won't work, with no ln following.

    layer = model.model.mm_projector
    bake_mean_into_linearmm(layer[0])
    # fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)]) # 
    # no post ln, just add rotate 
    
    model_utils.replace_modules(
        model.model.vision_tower.vision_tower.vision_model.encoder,
        torch.nn.LayerNorm,
        lambda _: model_utils.RMSLN(model.model.vision_tower.config.hidden_size),
        replace_layers=False,
    )


def fuse_layer_normsvit_return(model):
    
    model_type = model_utils.get_model_typevit(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion # why is this?
    # for W in model_utils.get_embeddings(**kwargs):
    #     W_ = W.weight.data.double()
    #     W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
    ########## VIT no embedding replace this to pre-LN
    # Replace Pre-LN in vit with substract mean
    vit_model = model.model.vision_tower.vision_tower.vision_model
    vit_model.pre_layrnorm = model_utils.LN_(vit_model.pre_layrnorm)
    layers = model_utils.get_vit_layers(**kwargs)
    Ws = []
    bs = []
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for idx, layer in enumerate(layers):
        
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAVA_MODEL:
            fuse_ln_linear(layer.layer_norm1, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.layer_norm2, [layer.mlp.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
        if model_type == model_utils.LLAVA_MODEL:
            # here do not include the second last layers
            w, b = bake_mean_into_linear_return(layer.self_attn.out_proj)
            if idx < len(layers) - 1:
                Ws.append(w+b)
                bs.append(b.unsqueeze(0))
            w, b = bake_mean_into_linear_return(layer.mlp.fc2) # here skip the second last one, won't work, with no ln following.
            if idx < len(layers) - 1:
                Ws.append((w+ b).reshape(-1, 1024) )
                bs.append(b.unsqueeze(0))
            # if  idx < len(layers) - 2:
            #     bake_mean_into_linear(layer.mlp.fc2) # here skip the second last one, won't work, with no ln following.
            # else:
            #     print(f"skip mlp mean in layer{idx}")
            #     pass
    # fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)]) # 
    # no post ln, just add rotate 
    
    model_utils.replace_modules(
        model.model.vision_tower.vision_tower.vision_model.encoder,
        torch.nn.LayerNorm,
        lambda _: model_utils.RMSLN(model.model.vision_tower.config.hidden_size),
        replace_layers=False,
    )
    return torch.cat(Ws, dim=0), torch.cat(bs, dim=0)


def fuse_layer_normsvit_returnskip(model):
    
    model_type = model_utils.get_model_typevit(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    ########## VIT no embedding replace this to pre-LN
    # Replace Pre-LN in vit with substract mean
    if type(model) == model_utils.LLAVA_NEXT_HF:
        vit_model = model.vision_tower.vision_model
    else:
        vit_model = model.model.vision_tower.vision_tower.vision_model
    vit_model.pre_layrnorm = model_utils.LN_(vit_model.pre_layrnorm)
    layers = model_utils.get_vit_layers(**kwargs)
    Ws = []
    bs = []
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for idx, layer in enumerate(layers):
        
        # fuse the input layernorms into the linear layers
        if model_type in [model_utils.LLAVA_MODEL, model_utils.LLAVA_NEXT_HF]:
            fuse_ln_linear(layer.layer_norm1, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.layer_norm2, [layer.mlp.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
        if model_type in [model_utils.LLAVA_MODEL, model_utils.LLAVA_NEXT_HF]:
            # here do not include the second last layers
            w, b = bake_mean_into_linear_return(layer.self_attn.out_proj)
            if  idx < len(layers) - 2:
                w, b = bake_mean_into_linear_return(layer.mlp.fc2) # here skip the second last one, won't work, with no ln following.
            else:
                logging.info(f"skip mlp mean in layer{idx}")
            # if  idx < len(layers) - 2:
            #     bake_mean_into_linear(layer.mlp.fc2) # here skip the second last one, won't work, with no ln following.
            # else:
            #     print(f"skip mlp mean in layer{idx}")
            #     pass
    # fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)]) # 
    # no post ln, just add rotate 
    
    if type(model) == model_utils.LLAVA_NEXT_HF:
        model_utils.replace_modules(
            model.vision_tower.vision_model.encoder,
            torch.nn.LayerNorm,
            lambda _: model_utils.RMSLN(model_utils.get_vit_config(model).hidden_size, eps=1e-5),
            replace_layers=False,
        )
    else:
        model_utils.replace_modules(
            model.model.vision_tower.vision_tower.vision_model.encoder,
            torch.nn.LayerNorm,
            lambda _: model_utils.RMSLN(model_utils.get_vit_config(model).hidden_size, eps=1e-5),
            replace_layers=False,
        )
    logging.info('finish vit ln fuse with eps 1e-5')
    return None


def fuse_layer_normsvit_returnskipsvd(model):
    
    model_type = model_utils.get_model_typevit(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion # why is this?
    # for W in model_utils.get_embeddings(**kwargs):
    #     W_ = W.weight.data.double()
    #     W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
    ########## VIT no embedding replace this to pre-LN
    # Replace Pre-LN in vit with substract mean
    vit_model = model.model.vision_tower.vision_tower.vision_model
    vit_model.pre_layrnorm = model_utils.LN_(vit_model.pre_layrnorm)
    layers = model_utils.get_vit_layers(**kwargs)
    Ws = []
    bs = []
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for idx, layer in enumerate(layers):
        
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAVA_MODEL:
            fuse_ln_linear(layer.layer_norm1, [layer.self_attn.q_proj.BLinear, layer.self_attn.k_proj.BLinear, layer.self_attn.v_proj.BLinear])
            fuse_ln_linear(layer.layer_norm2, [layer.mlp.fc1.BLinear])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
        if model_type == model_utils.LLAVA_MODEL:
            # here do not include the second last layers
            w, b = bake_mean_into_linear_return(layer.self_attn.out_proj.ALinear)
            if idx < len([]) - 1:
                Ws.append(w+b)
                bs.append(b.unsqueeze(0))
            if  idx < len(layers) - 2:
                w, b = bake_mean_into_linear_return(layer.mlp.fc2.ALinear) # here skip the second last one, won't work, with no ln following.
                if idx < len([]) - 1:
                    Ws.append((w+ b).reshape(-1, 1024) )
                    bs.append(b.unsqueeze(0))
            else:
                logging.info(f"skip mlp mean in layer{idx}")
            # if  idx < len(layers) - 2:
            #     bake_mean_into_linear(layer.mlp.fc2) # here skip the second last one, won't work, with no ln following.
            # else:
            #     print(f"skip mlp mean in layer{idx}")
            #     pass
    # fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)]) # 
    # no post ln, just add rotate 
    
    model_utils.replace_modules(
        model.model.vision_tower.vision_tower.vision_model.encoder,
        torch.nn.LayerNorm,
        lambda _: model_utils.RMSLN(model.model.vision_tower.config.hidden_size),
        replace_layers=False,
    )
    return None


def fuse_layer_normsvit_skip(model):
    
    model_type = model_utils.get_model_typevit(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion # why is this?
    # for W in model_utils.get_embeddings(**kwargs):
    #     W_ = W.weight.data.double()
    #     W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
    ########## VIT no embedding replace this to pre-LN
    # Replace Pre-LN in vit with substract mean
    vit_model = model.model.vision_tower.vision_tower.vision_model
    vit_model.pre_layrnorm = model_utils.LN_(vit_model.pre_layrnorm)
    layers = model_utils.get_vit_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for idx, layer in enumerate(layers):
        
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAVA_MODEL:
            fuse_ln_linear(layer.layer_norm1, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.layer_norm2, [layer.mlp.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
        if model_type == model_utils.LLAVA_MODEL:
            # here do not include the second last layers
            bake_mean_into_linear(layer.self_attn.out_proj)
            if  idx < len(layers) - 2:
                bake_mean_into_linear(layer.mlp.fc2) # here skip the second last one, won't work, with no ln following.
            else:
                logging.info(f"skip mlp mean in layer{idx}")

    
    # fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)]) # 
    # no post ln, just add rotate 
    
    model_utils.replace_modules(
        model.model.vision_tower.vision_tower.vision_model.encoder,
        torch.nn.LayerNorm,
        lambda _: model_utils.RMSLN(model.model.vision_tower.config.hidden_size),
        replace_layers=False,
    )


def wrap_layer_normvit(model):
    model_type = model_utils.get_model_typevit(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    layers = model_utils.get_vit_layers(**kwargs)
    for idx, layer in enumerate(layers):
        layer.layer_norm1 = model_utils.LNRotWrapper(layer.layer_norm1)
        layer.layer_norm2 = model_utils.LNRotWrapper(layer.layer_norm2)

    
def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

def get_orthogonal_matrix(size, mode, seed=42, device=utils.get_dev()):
    utils.set_seed(seed)
        
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')

    
def _rotate_embedding_matrix(param: torch.nn.Parameter, Q: torch.Tensor) -> None:
    dtype = param.data.dtype
    param_ = param.data.to(device=utils.get_dev(), dtype=torch.float64)
    param.data = torch.matmul(param_, Q).to(device="cpu", dtype=dtype)


def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    for W in model_utils.get_embeddings(model, model_type):
        if hasattr(W, "weight"):
            _rotate_embedding_matrix(W.weight, Q)
        elif hasattr(W, "embedding"):
            _rotate_embedding_matrix(W.embedding, Q)
            if hasattr(W, "new_embedding"):
                _rotate_embedding_matrix(W.new_embedding, Q)
        else:
            raise AttributeError(f"Cannot locate embedding weights on {type(W)}")


def rotate_projectors(model, Q: torch.Tensor) -> None:
    # Rotate the projectors.
    from .quant_utils import ActQuantWrapper
    model_type = model_utils.model_type_extractor(model)
    for proj in model_utils.get_projector(model, model_type):
        if model_type == model_utils.MOLMO_MODEL:
            targets = []
            modules = list(proj) if isinstance(proj, nn.ModuleList) else [proj]
            for module in modules:
                module = getattr(module, "submodule", module)
                if hasattr(module, "w2"):
                    targets.append(module.w2)
                elif isinstance(module, nn.Linear):
                    targets.append(module)
            for W in targets:
                dtype = W.weight.data.dtype
                W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
                W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
                if W.bias is not None:
                    b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
                    W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
            continue
        if type(proj) in [nn.Linear, ActQuantWrapper]:
            W = proj # for smolvlm
        elif model_type == model_utils.LLAVA_NEXT_HF:
            W = proj.linear_2
        else:
            W = proj[-1]
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        if W.bias is not None:
            b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
    
def rotate_projectorssvd(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    for proj in model_utils.get_projector(model, model_type):
        if type(proj) in [nn.Linear]:
            W = proj
        else:
            W = proj[-1].ALinear
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        if W.bias is not None:
            b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_attention_inputssvdkv(layer, Q, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj.BLinear, layer.self_attn.v_proj.BLinear]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_attention_inputssvdqkv(layer, Q, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj.BLinear, layer.self_attn.k_proj.BLinear, layer.self_attn.v_proj.BLinear]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_attention_inputs(layer, Q, model_type, svd_modules=None) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    if svd_modules in ['all', 'qkv', 'attn']:
        for W in [layer.self_attn.q_proj.BLinear, layer.self_attn.k_proj.BLinear, layer.self_attn.v_proj.BLinear]:
            dtype = W.weight.dtype
            W_ = W.weight.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    else:
        for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
            dtype = W.weight.dtype
            W_ = W.weight.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_attention_inputs_localft(layer, Q, model_type, args, svd_modules=None) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    # if svd_modules in ['all', 'qkv', 'attn']:
    #     for W in [layer.self_attn.q_proj.BLinear, layer.self_attn.k_proj.BLinear, layer.self_attn.v_proj.BLinear]:
    #         dtype = W.weight.dtype
    #         W_ = W.weight.to(device=utils.get_dev(), dtype=torch.float64)
    #         W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    # else:
    ###### U, V shape:
        # is_perhead
            # U: n, c, c(r)
            # S: n, C(r)
            # V: n, C_in, c(r)
        # original
            # U: Cout, C(r)
            # S: C(r)
            # V: C_in, C(r)
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        if args.svd_modules in ['all', 'qkv', 'attn']:
            V = W.svd_info_before_rot['V']
            if args.is_per_head_svd:
                if isinstance(V, list):
                    v_dtype = V[0].dtype
                    device = V[0].device
                    V = [(Q.t() @ vv.to(device=utils.get_dev(), dtype=torch.float64)).to(v_dtype).to(device) for vv in V]
                else:
                    v_dtype = V.dtype
                    device = V.device
                    V = (Q.t() @ V.to(device=utils.get_dev(), dtype=torch.float64)).to(v_dtype).to(device)
            else:
                v_dtype = V.dtype
                device = V.device
                V = (Q.t() @ V.to(device=utils.get_dev(), dtype=torch.float64)).to(v_dtype).to(device)
            W.svd_info_before_rot['V'] = V
        W_ = W.weight.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        

def rotate_attention_qk(layer, Q, model_type) -> None:
    # do not need to rotate vit q k matrix, as no kv cache needed.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        if W.bias is not None:
            b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_attention_output(layer, Q, model_type, svd_modules=None) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type in [model_utils.LLAMA_MODEL, model_utils.SMOVLM_MODEL, model_utils.LLAVA_NEXT_HF, model_utils.MOLMO_MODEL] or _is_qwen2vl_model_type(model_type):
        if svd_modules in ['attn', 'all']: # oproj svd is enabled for attn/all mode
            W = layer.self_attn.o_proj.ALinear
        else:
            W = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL or model_type == model_utils.LLAVA_MODEL:
        W = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_attention_output_localft(layer, Q, model_type, args, svd_modules=None) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type in [model_utils.LLAMA_MODEL, model_utils.SMOVLM_MODEL, model_utils.LLAVA_NEXT_HF, model_utils.MOLMO_MODEL] or _is_qwen2vl_model_type(model_type):
        W = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL or model_type == model_utils.LLAVA_MODEL:
        W = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    if args.svd_modules in ['all', 'o', 'attn']:
        U = W.svd_info_before_rot['U'] # Cout, C(r)
        u_dtype = U.dtype
        device = U.device
        U = U.to(device=utils.get_dev(), dtype=torch.float64)
        U = torch.matmul(Q.T, U).to(device=device, dtype=u_dtype)
        W.svd_info_before_rot['U'] = U
    W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, Q, model_type, svd_modules=None):
    # Rotate the MLP input weights.
    if _is_llama_style_qwen2vl_lm(model_type):
        if svd_modules in ['all', 'mlp', 'gaup']:
            mlp_inputs = [layer.mlp.up_proj.BLinear, layer.mlp.gate_proj.BLinear]
        else:
            mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.OPT_MODEL:
        mlp_inputs = [layer.fc1]
    elif model_type == model_utils.LLAVA_MODEL:
        mlp_inputs = [layer.mlp.fc1]
    elif model_type == model_utils.SMOVLM_MODEL:
        mlp_inputs = [layer.mlp.fc1]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_mlp_input_localft(layer, Q, model_type, args, svd_modules=None):
    # Rotate the MLP input weights.
    if _is_llama_style_qwen2vl_lm(model_type):
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.OPT_MODEL:
        mlp_inputs = [layer.fc1]
    elif model_type == model_utils.LLAVA_MODEL:
        mlp_inputs = [layer.mlp.fc1]
    elif model_type == model_utils.SMOVLM_MODEL:
        mlp_inputs = [layer.mlp.fc1]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        if args.svd_modules in ['all', 'mlp', 'gaup']:
            V = W.svd_info_before_rot['V']
            v_dtype = V.dtype
            device = V.device
            V = V.to(device=utils.get_dev(), dtype=torch.float64)
            V = torch.matmul(Q.T, V).to(device=device, dtype=v_dtype)
            W.svd_info_before_rot['V'] = V
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_mlp_inputlm(layer, Q, model_type):
    # Rotate the MLP input weights.
    if model_type == model_utils.LLAMA_MODEL or _is_qwen2vl_model_type(model_type):
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.OPT_MODEL:
        mlp_inputs = [layer.fc1]
    elif model_type == model_utils.LLAVA_MODEL:
        mlp_inputs = [layer.mlp.fc1]
    elif model_type == model_utils.LLAMAV_MODLE:
        mlp_inputs = [layer.mlp.fc1]
    elif model_type == model_utils.SMOVLM_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
   
def rotate_mlp_output(layer, Q, model_type, svd_modules=None):
    # Rotate the MLP output weights and bias.
    if model_type in [model_utils.LLAMA_MODEL, model_utils.SMOVLM_MODEL, model_utils.LLAVA_NEXT_HF, model_utils.MOLMO_MODEL] or _is_qwen2vl_model_type(model_type):
        if svd_modules in ['all', 'mlp', 'down']:
            W = layer.mlp.down_proj.ALinear
            Wb = layer.mlp.down_proj.BLinear
        else:
            W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    elif model_type == model_utils.LLAVA_MODEL:
        W = layer.mlp.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if svd_modules in ['all', 'mlp']:
        apply_exact_had_to_linear(Wb, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp down proj SVD.BLinearinput
    else:
        apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp dowm proj input
    if W.bias is not None:
        b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_output_hybrid_ffn(layer, Q, model_type):
    """Rotate only the residual-stream side of the FFN output projection.

    Hybrid FFN later permutes the FFN hidden channels, so the standard QuaRot
    Hadamard pair on the FFN hidden dimension would no longer commute with that
    grouping. Keep the Q rotation on the output side, but intentionally skip the
    FFN-hidden Hadamard absorption/runtime pair for the hybrid path.
    """
    if model_type in [model_utils.LLAMA_MODEL, model_utils.SMOVLM_MODEL, model_utils.LLAVA_NEXT_HF, model_utils.MOLMO_MODEL] or _is_qwen2vl_model_type(model_type):
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    elif model_type == model_utils.LLAVA_MODEL:
        W = layer.mlp.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_output_localft(layer, Q, model_type, args, svd_modules=None):
    # Rotate the MLP output weights and bias.
    if model_type in [model_utils.LLAMA_MODEL, model_utils.SMOVLM_MODEL, model_utils.LLAVA_NEXT_HF, model_utils.MOLMO_MODEL] or _is_qwen2vl_model_type(model_type):
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    elif model_type == model_utils.LLAVA_MODEL:
        W = layer.mlp.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    # U: Cout, C(r)
    # S: C(r)
    # V: C_in, C(r)
    # output
    W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if args.svd_modules in ['all', 'mlp', 'down']:
        U = W.svd_info_before_rot['U']
        u_dtype = U.dtype
        device = U.device
        U = U.to(device=utils.get_dev(), dtype=torch.float64)
        U = torch.matmul(Q.T, U).to(device=device, dtype=u_dtype)
        W.svd_info_before_rot['U'] = U

    # input 
    apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp dowm proj input
    if W.bias is not None:
        b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
    if args.svd_modules in ['all', 'mlp', 'down']:
        V = W.svd_info_before_rot['V']
        v_dtype = V.dtype
        device = V.device
        V = V.to(device=utils.get_dev(), dtype=torch.float64)
        V = apply_exact_had_to_UV(V, had_dim=-1, output=True)
        W.svd_info_before_rot['V'] = V.to(device=device, dtype=v_dtype)


def apply_exact_had_to_UV(UV, had_dim=-1, output=False):
    from .hadamard_utils import get_hadK, matmul_hadU_cuda
    # note: should always pass UV as a single matrix (out, in)
    device = UV.device
    dtype = UV.dtype
    UV = UV.float().cuda()
    if had_dim == -1:
        if output:
            out_features = UV.shape[-2]
            had_K, K = get_hadK(out_features)
            UV = matmul_hadU_cuda(UV.t(), had_K, K).t()
        else:
            in_features = UV.shape[-1]
            had_K, K = get_hadK(in_features)
            UV = matmul_hadU_cuda(UV, had_K, K)
    return UV.to(device=device, dtype=dtype)

def rotate_mlp_output_noonline(layer, Q, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    elif model_type == model_utils.LLAVA_MODEL:
        W = layer.mlp.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_mlp_outputvit(layer, Q, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    elif model_type == model_utils.LLAVA_MODEL:
        W = layer.mlp.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def matmul_hadU_cuda_had(X, hadK, transpose=False):
    '''
    Apply hadamard transformation. 
    It reshapes X and applies Walsh-Hadamard transform to the last dimension. 
    Then, it will multiply the retult by another hadamard matrix.
    '''
    from fast_hadamard_transform import hadamard_transform
    from .hadamard_utils import get_had172
    n = X.shape[-1]
    K = hadK.shape[-1]

    if transpose:
        hadK = hadK.T.contiguous()
    input = X.float().cuda().view(-1, K, n // K)
    input = hadamard_transform(input.contiguous(), scale=1/math.sqrt(n))
    input = hadK.to(input.device).to(input.dtype) @ input 
    return input.to(X.device).to(X.dtype).reshape(
        X.shape) 

def rotate_faster_down_proj(layer, model_type, hardK):
    from fast_hadamard_transform import hadamard_transform
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    else:
        raise ValueError(f'Faster MLP is onlu supported for LLaMa models!')
    
    dtype = W.weight.data.dtype
    W.weight.data = matmul_hadU_cuda_had(W.weight.data.float().cuda(), hardK)
    W.weight.data = W.weight.data.to(device="cpu", dtype=dtype)

def rotate_projectorsvit(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    for proj in model_utils.get_projector(model, model_type):
        if type(proj) in [nn.Linear]:
            W = proj
        else:
            W = proj[0]
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
  
def rotate_projectorsinput(model, Q=None) -> None:
    model_type = model_utils.model_type_extractor(model)
    for proj in model_utils.get_projector(model, model_type):
        if type(proj) in [nn.Linear]:
            W = [proj]
        elif model_type == model_utils.LLAVA_NEXT_HF:
            W = [proj.linear_1, proj.linear_2]
        else:
            W = [proj[0], proj[2]]
        for W_ in W:
            if model_type == model_utils.SMOVLM_MODEL:
                dtype = W_.weight.data.dtype
                W__ = W_.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
                init_shape = W__.shape
                W__ = W__.reshape((-1, W__.shape[-1]//1152, 1152))
                W_.weight.data = torch.matmul(W__, Q).reshape(init_shape).to(device="cpu", dtype=dtype)
            else:
                apply_exact_had_to_linear(W_, had_dim=-1, output=False)


def rotate_projectorsinput_mmR(model, Q) -> None:
    model_type = model_utils.model_type_extractor(model)
    for proj in model_utils.get_projector(model, model_type):
        if model_type == model_utils.LLAVA_NEXT_HF:
            W_ = proj.linear_1
            apply_exact_had_to_linear(W_, had_dim=-1, output=False)
            W = proj.linear_2
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        else:
            W_ = proj[0]
            apply_exact_had_to_linear(W_, had_dim=-1, output=False)
            W = proj[2]
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_projectorsinputllava_fused(model, Q: torch.Tensor) -> None:
    model_type = model_utils.model_type_extractor(model)
    for proj in model_utils.get_projector(model, model_type):
        if model_type == model_utils.LLAVA_NEXT_HF:
            W = proj.linear_1
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
            W_ = proj.linear_2
            apply_exact_had_to_linear(W_, had_dim=-1, output=False)
        else:
            W = proj[0]
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
            W_ = proj[2]
            apply_exact_had_to_linear(W_, had_dim=-1, output=False)

def rotate_projectorsinputllava_fused_mmR(model, Q: torch.Tensor, Q1: torch.Tensor,) -> None:
    model_type = model_utils.model_type_extractor(model)
    for proj in model_utils.get_projector(model, model_type):
        if model_type == model_utils.LLAVA_NEXT_HF:
            W = proj.linear_1
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
            W = proj.linear_2
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q1).to(device="cpu", dtype=dtype)
        else:
            W = proj[0]
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
            W = proj[2]
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q1).to(device="cpu", dtype=dtype)

def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model, model_type=model_utils.model_type_extractor(model))
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def _apply_exact_had_to_bias(bias: torch.nn.Parameter, had_dim=-1) -> None:
    if bias is None:
        return
    dtype = bias.data.dtype
    dev = bias.data.device
    b = bias.data.float().cuda()
    init_shape = b.shape
    if had_dim == -1:
        had_K, K = hadamard_utils.get_hadK(init_shape[-1])
        b = hadamard_utils.matmul_hadU_cuda(b, had_K, K)
    else:
        b = hadamard_transform(
            b.reshape(-1, init_shape[-1] // had_dim, had_dim),
            scale=1 / math.sqrt(had_dim),
        ).reshape(init_shape)
    bias.data = b.to(device=dev, dtype=dtype)


def rotate_ov_proj(layer, model_type, head_num, head_dim, args, svd_modules=None):
    v_proj_module = layer.self_attn.v_proj
    if svd_modules in ['all', 'attn', 'qkv']:
        v_proj = v_proj_module.ALinear
    else:
        v_proj = v_proj_module
    if model_type in [model_utils.LLAMA_MODEL, model_utils.SMOVLM_MODEL, model_utils.LLAVA_NEXT_HF, model_utils.MOLMO_MODEL] or _is_qwen2vl_model_type(model_type):
        if svd_modules in ['attn', 'all']:
            o_proj = layer.self_attn.o_proj.BLinear
        else:
            o_proj = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL or model_type == model_utils.LLAVA_MODEL:
        o_proj = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
    if args.is_per_head_svd and isinstance(v_proj, nn.ParameterList):
        for i, vv in enumerate(v_proj):
            v_dtype = vv.dtype
            device = vv.device
            new_v = apply_exact_had_to_UV(
                vv.to(device=utils.get_dev(), dtype=torch.float64), had_dim=-1, output=True
            ).to(device=device, dtype=v_dtype)
            v_proj[i].data.copy_(new_v) # inplace update
    elif args.is_per_head_svd and isinstance(v_proj, (list, tuple)):
        v_dtype = v_proj[0].dtype
        device = v_proj[0].device
        v_proj = [apply_exact_had_to_UV(vv.to(device=utils.get_dev(), dtype=torch.float64), had_dim=-1, output=True).to(device=device, dtype=v_dtype) for vv in v_proj]
        v_proj = torch.stack(v_proj, dim=0)
        layer.self_attn.v_proj.ALinear = v_proj
    elif args.is_per_head_svd and isinstance(v_proj, nn.Parameter):
        # Stacked per-head ALinear [num_heads, head_dim, head_rank] — same Hadamard as nn.Linear(out=H*D, in=r).
        H, D, r = v_proj.shape
        lin = nn.Linear(r, H * D, bias=False).to(device=v_proj.device, dtype=v_proj.dtype)
        lin.weight.data.copy_(v_proj.data.reshape(H * D, r))
        apply_exact_had_to_linear(lin, had_dim=head_dim, output=True)
        v_proj.data.copy_(lin.weight.data.view(H, D, r))
    else:
        # Pre-SVD path (or non per-head SVD): v_proj is a standard nn.Linear.
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    if (
        getattr(args, "is_per_head_svd", False)
        and svd_modules in ['all', 'attn', 'qkv']
        and hasattr(v_proj_module, "bias")
        and not isinstance(v_proj, nn.Linear)
    ):
        _apply_exact_had_to_bias(v_proj_module.bias, had_dim=head_dim)

    apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)

def rotate_ov_proj_localft(layer, model_type, head_num, head_dim, args, svd_modules=None):    
    v_proj = layer.self_attn.v_proj
    if model_type in [model_utils.LLAMA_MODEL, model_utils.SMOVLM_MODEL, model_utils.LLAVA_NEXT_HF, model_utils.MOLMO_MODEL] or _is_qwen2vl_model_type(model_type):
        o_proj = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL or model_type == model_utils.LLAVA_MODEL:
        o_proj = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True) # output
    if args.svd_modules in ['all', 'qkv', 'attn']:
        U = v_proj.svd_info_before_rot['U']
        
        
        # U = U.
        if args.is_per_head_svd:
            if isinstance(U, list):
                u_dtype = U[0].dtype
                device = U[0].device
                U = [apply_exact_had_to_UV(uu.to(device=utils.get_dev(), dtype=torch.float64), had_dim=-1, output=True).to(device=device, dtype=u_dtype) for uu in U]
            else:
                u_dtype = U.dtype
                device = U.device
                U = [apply_exact_had_to_UV(uu.to(device=utils.get_dev(), dtype=torch.float64), had_dim=-1, output=True).to(device=device, dtype=u_dtype) for uu in U]
                U = torch.stack(U, dim=0)
        else:
            logging.info('Hadamard apply_exact not supported for none-per head SVD')
        v_proj.svd_info_before_rot['U'] = U

    apply_exact_had_to_linear(o_proj, had_dim=-1, output=False) # input
    if args.svd_modules in ['all', 'o', 'attn']:
        V = o_proj.svd_info_before_rot['V']
        v_dtype = V.dtype
        device = V.device
        V = V.to(device=utils.get_dev(), dtype=torch.float64)
        V = apply_exact_had_to_UV(V, had_dim=-1, output=True) # here V will be transposed, so output=True
        o_proj.svd_info_before_rot['V'] = V.to(device=device, dtype=v_dtype)


def rotate_ov_projvit(layer, model_type, head_num, head_dim):
    v_proj = layer.self_attn.v_proj
    if model_type == model_utils.LLAMA_MODEL:
        o_proj = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL or model_type == model_utils.LLAVA_MODEL or model_type == model_utils.SMOVLM_MODEL:
        o_proj = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
    if model_type == model_utils.SMOVLM_MODEL:
        apply_exact_had_to_linear_nopow2(v_proj, had_dim=head_dim, output=True)
        apply_exact_had_to_linear_nopow2(o_proj, had_dim=-1, output=False)
    else:
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
        apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)

@torch.inference_mode()
def rotate_model(model, args):
    Q = get_orthogonal_matrix(model.config.hidden_size,
                                                args.rotate_mode, args.seed)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, Q) # +rotate_projector
    rotate_head(model, Q) #
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, 
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type, args.svd_modules)
        rotate_attention_output(layers[idx], Q, model_type, args.svd_modules)
        rotate_mlp_input(layers[idx], Q, model_type, args.svd_modules)
        rotate_mlp_output(layers[idx], Q, model_type, args.svd_modules) # no in vit
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim, args, args.svd_modules)

@torch.inference_mode()
def rotate_modelllava(model, args):
    model_type = model_utils.model_type_extractor(model)
    config = _get_lm_config(model, model_type)
    Q = get_orthogonal_matrix(config.hidden_size,
                                                args.rotate_mode, args.seed)
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    rotate_embeddings(model, Q) # +rotate_projector
    rotate_projectors(model, Q) # +rotate_projector
    rotate_head(model, Q) # is lm head
    utils.cleanup_memory()
    layers = _get_transformer_layers(model, model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_output(layers[idx], Q, model_type) # no in vit
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim, args)
    _wrap_molmo_post_norms_for_rotation(model, model_type, Q)


@torch.inference_mode()
def rotate_modelllava_localft(model, args):
    model_type = model_utils.model_type_extractor(model)
    config = _get_lm_config(model, model_type)
    Q = get_orthogonal_matrix(config.hidden_size,
                                                args.rotate_mode, args.seed)
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    rotate_embeddings(model, Q) # +rotate_projector
    rotate_projectors(model, Q) # +rotate_projector
    rotate_head(model, Q) # is lm head
    utils.cleanup_memory()
    layers = _get_transformer_layers(model, model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs_localft(layers[idx], Q, model_type, args)
        rotate_attention_output_localft(layers[idx], Q, model_type, args)
        rotate_mlp_input_localft(layers[idx], Q, model_type, args)
        rotate_mlp_output_localft(layers[idx], Q, model_type, args) # no in vit
        rotate_ov_proj_localft(layers[idx], model_type, num_heads, head_dim, args)
    _wrap_molmo_post_norms_for_rotation(model, model_type, Q)


@torch.inference_mode()
def rotate_modelsmolvlm(model, args):
    Q = get_orthogonal_matrix(model.config.text_config.hidden_size,
                                                args.rotate_mode, args.seed)
    config = model.config.text_config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, Q) # +rotate_projector
    rotate_projectors(model, Q) # +rotate_projector
    rotate_head(model, Q) # is lm head
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, 
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_inputlm(layers[idx], Q, model_type)
        rotate_mlp_output(layers[idx], Q, model_type) # no in vit
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim, args)


@torch.inference_mode()
def rotate_modelllavasvd(model, args):
    Q = get_orthogonal_matrix(model.config.hidden_size,
                                                args.rotate_mode, args.seed)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, Q) # +rotate_projector
    rotate_projectors(model, Q) # +rotate_projector
    rotate_head(model, Q) # is lm head
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, 
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_output_noonline(layers[idx], Q, model_type) # no in vit
    _wrap_molmo_post_norms_for_rotation(model, model_type, Q)

@torch.inference_mode()
def rotate_modelllavasvdkv(model, args):
    Q = get_orthogonal_matrix(model.config.hidden_size,
                                                args.rotate_mode, args.seed)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, Q) # +rotate_projector
    rotate_projectors(model, Q) # +rotate_projector
    rotate_head(model, Q) # is lm head
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, 
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputssvdkv(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_output(layers[idx], Q, model_type) # no in vit
        rotate_ov_projsvdkv(layers[idx], model_type, num_heads, head_dim)
    _wrap_molmo_post_norms_for_rotation(model, model_type, Q)

@torch.inference_mode()
def rotate_modelllavasvdqkv(model, args):
    model_type = model_utils.model_type_extractor(model)
    config = _get_lm_config(model, model_type)
    Q = get_orthogonal_matrix(config.hidden_size,
                                                args.rotate_mode, args.seed)
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    rotate_embeddings(model, Q) # +rotate_projector
    rotate_projectors(model, Q) # +rotate_projector
    rotate_head(model, Q) # is lm head
    utils.cleanup_memory()
    layers = _get_transformer_layers(model, model_type=model_type)
    
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputssvdqkv(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type, args.svd_modules)
        rotate_mlp_input(layers[idx], Q, model_type, args.svd_modules)
        if getattr(args, "use_hybrid_ffn", False):
            rotate_mlp_output_hybrid_ffn(layers[idx], Q, model_type)
        else:
            rotate_mlp_output(layers[idx], Q, model_type, args.svd_modules) # no in vit
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim, args, args.svd_modules)
    _wrap_molmo_post_norms_for_rotation(model, model_type, Q)
        
@torch.inference_mode()
def rotate_modelsmolvlmsvdqkv(model, args):
    Q = get_orthogonal_matrix(model.config.text_config.hidden_size,
                                                args.rotate_mode, args.seed)
    config = model.config.text_config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, Q) # +rotate_projector
    rotate_projectors(model, Q) # +rotate_projector
    rotate_head(model, Q) # is lm head
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, 
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputssvdqkv(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type, args.svd_modules)
        rotate_mlp_inputlm(layers[idx], Q, model_type)
        rotate_mlp_output(layers[idx], Q, model_type, args.svd_modules) # no in vit
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim, args, args.svd_modules)


@torch.inference_mode()
def rotate_modelllavaWout(model, args, rotations):
    R1 = rotations['R1'].double()
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, R1) # +rotate_projector
    rotate_projectors(model, R1) # +rotate_projector
    rotate_head(model, R1) # is lm head
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, 
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], R1, model_type)
        rotate_attention_output(layers[idx], R1, model_type)
        rotate_mlp_input(layers[idx], R1, model_type)
        rotate_mlp_output(layers[idx], R1, model_type) # no in vit
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim, args)
        

@torch.inference_mode()
def rotate_modelvit(model, args):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode, args.seed)
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    ln = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    ln.online_random_had = True
    ln.had_K = Q
    ln.K = 1
    ln.fp32_had = args.fp32_had #
    model_type = model_utils.get_model_typevit(model)
    rotate_projectorsvit(model, Q) # rotate projector input 
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating vit layers")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_outputvit(layers[idx], Q, model_type) # no in vit
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)


@torch.inference_mode()
def rotate_modelvitmm(model, args):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode, args.seed)
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    ln = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    ln.online_random_had = True
    ln.had_K = Q
    ln.K = 1
    ln.fp32_had = args.fp32_had #
    model_type = model_utils.get_model_typevit(model)
    rotate_projectorsvit(model, Q) # +rotate_projector
    rotate_projectorsinput(model) # here have mm proj issue
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating vit mm")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_outputvit(layers[idx], Q, model_type) # no in vit
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
    

@torch.inference_mode()
def rotate_modelvitmmv2(model, args):
    Q = get_orthogonal_matrix(model_utils.get_vit_config(model).hidden_size,
                                                args.rotate_mode, args.seed)
    Q1 = get_orthogonal_matrix(model_utils.get_vit_config(model).intermediate_size,
                                                args.rotate_mode)
    if type(model) == model_utils.LLAVA_NEXT_HF:
        Q2 = get_orthogonal_matrix(model.config.hidden_size, # ?
                                                args.rotate_mode, args.seed) if args.mm_rh else None
    else:
        Q2 = get_orthogonal_matrix(model.model.config.hidden_size,
                                                args.rotate_mode) if args.mm_rh else None
    config = model_utils.get_vit_config(model)
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    if type(model) == model_utils.LLAVA_NEXT_HF:
        ln = model.vision_tower.vision_model.pre_layrnorm
    else:
        ln = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    ln.online_random_had = True
    ln.had_K = Q
    ln.K = 1
    ln.fp32_had = args.fp32_had #
    model_type = model_utils.get_model_typevit(model)
    if not args.vit_mmoff:
        if args.mm_rh:
            rotate_projectorsinputllava_fused_mmR(model, Q, Q2)
        else:
            rotate_projectorsinputllava_fused(model, Q)
    else:
        rotate_projectorsvit(model, Q)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating vit mm")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_outputvit(layers[idx], Q, model_type) # no mlpfc2 online in vit
        W = layer.mlp.fc2
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q1).to(device="cpu", dtype=dtype)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
    if args.mm_rh:
        return Q1, Q2
    return Q1


@torch.inference_mode()
def rotate_modelvitmmv2svd(model, args):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode, args.seed)
    Q1 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                args.rotate_mode)
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    ln = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    ln.online_random_had = True
    ln.had_K = Q
    ln.K = 1
    ln.fp32_had = args.fp32_had #
    model_type = model_utils.get_model_typevit(model)
    rotate_projectorsinputllava_fusedsvd(model, Q)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating vit mm")):
        rotate_attention_inputssvd(layers[idx], Q, model_type)
        rotate_attention_outputsvd(layers[idx], Q, model_type)
        rotate_mlp_inputsvd(layers[idx], Q, model_type)
        rotate_mlp_outputvitsvd(layers[idx], Q, model_type) # no mlpfc2 online in vit
        W = layer.mlp.fc2.BLinear
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q1).to(device="cpu", dtype=dtype)
        rotate_ov_projvitsvd(layers[idx], model_type, num_heads, head_dim)
    return Q1


@torch.inference_mode()
def rotate_modelvitmmv2learn(model, args, rotations):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode, args.seed)
    Q1 = rotations['R4'].double()
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    ln = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    ln.online_random_had = True
    ln.had_K = Q
    ln.K = 1
    ln.fp32_had = args.fp32_had #
    model_type = model_utils.get_model_typevit(model)
    rotate_projectorsinputllava_fused(model, Q)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating vit mm")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_outputvit(layers[idx], Q, model_type) # no mlpfc2 online in vit
        W = layer.mlp.fc2
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q1).to(device="cpu", dtype=dtype)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
    return Q1


@torch.inference_mode()
def rotate_modelvitmmv3(model, args, rotations):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode, args.seed)
    R3, R4 = rotations['R3'].double(), rotations['R4'].double()
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    ln = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    ln.online_random_had = True
    ln.had_K = Q
    ln.K = 1
    ln.fp32_had = args.fp32_had #
    model_type = model_utils.get_model_typevit(model)
    rotate_projectorsinputllava_fused(model, Q)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating vit mm")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_outputvit(layers[idx], Q, model_type) # no mlpfc2 online in vit
        # Later fused to function
        W = layer.mlp.fc1
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(R3.T, W_).to(device="cpu", dtype=dtype)
        if W.bias is not None:
            b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.bias.data = torch.matmul(R3.T, b).to(device="cpu", dtype=dtype)
        W = layer.mlp.fc2
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R4).to(device="cpu", dtype=dtype)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)


@torch.inference_mode()
def rotate_modelvitmmv4(model, args, rotations):
    # learnt wout Q
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode, args.seed)
    Q1 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                args.rotate_mode)
    R1 = rotations['R1'].double()
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    ln = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    ln.online_random_had = True
    ln.had_K = R1
    ln.K = 1
    ln.fp32_had = args.fp32_had #
    model_type = model_utils.get_model_typevit(model)
    rotate_projectorsinputllava_fused(model, R1)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating vit mm")):
        rotate_attention_inputs(layers[idx], R1, model_type)
        rotate_attention_output(layers[idx], R1, model_type)
        rotate_mlp_input(layers[idx], R1, model_type)
        rotate_mlp_outputvit(layers[idx], R1, model_type) # no mlpfc2 online in vit
        W = layer.mlp.fc2
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q1).to(device="cpu", dtype=dtype)
        # Later fused to function
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
    return Q1

@torch.inference_mode()
def rotate_modelvitonline(model, args):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode, args.seed)
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.get_model_typevit(model)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
            apply_exact_had_to_linear(W, had_dim=-1, output=False)
        W = layer.mlp.fc1
        apply_exact_had_to_linear(W, had_dim=-1, output=False)
        W = layer.mlp.fc2
        apply_exact_had_to_linear(W, had_dim=-1, output=False)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
        had_K, K = hadamard_utils.get_hadK(model_dim)
        layer.layer_norm1.online_full_had = True
        layer.layer_norm1.had_K = had_K
        layer.layer_norm1.K = K
        layer.layer_norm1.fp32_had = args.fp32_had
        layer.layer_norm2.online_full_had = True
        layer.layer_norm2.had_K = had_K
        layer.layer_norm2.K = K
        layer.layer_norm2.fp32_had = args.fp32_had


@torch.inference_mode()
def rotate_modelvitonlineR(model, args):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode, args.seed)
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.get_model_typevit(model)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type) # write new mlp_input for online rotation here
        W = layer.mlp.fc2
        apply_exact_had_to_linear(W, had_dim=-1, output=False)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
        layer.layer_norm1.online_random_had = True
        layer.layer_norm1.had_K = Q
        layer.layer_norm1.K = 1
        layer.layer_norm1.fp32_had = args.fp32_had
        layer.layer_norm2.online_random_had = True
        layer.layer_norm2.had_K = Q
        layer.layer_norm2.K = 1
        layer.layer_norm2.fp32_had = args.fp32_had

@torch.inference_mode()
def rotate_modelvitmmonline(model, args):
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.get_model_typevit(model)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
            apply_exact_had_to_linear(W, had_dim=-1, output=False)
        W = layer.mlp.fc1
        apply_exact_had_to_linear(W, had_dim=-1, output=False)
        W = layer.mlp.fc2
        apply_exact_had_to_linear(W, had_dim=-1, output=False)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
        had_K, K = hadamard_utils.get_hadK(model_dim)
        layer.layer_norm1.online_full_had = True
        layer.layer_norm1.had_K = had_K
        layer.layer_norm1.K = K
        layer.layer_norm1.fp32_had = args.fp32_had
        layer.layer_norm2.online_full_had = True
        layer.layer_norm2.had_K = had_K
        layer.layer_norm2.K = K
        layer.layer_norm2.fp32_had = args.fp32_had
    # add rotate projector layers, should be online
    rotate_projectorsinput(model)


@torch.inference_mode()
def rotate_modelvitmmonlineR(model, args):
    Q = get_orthogonal_matrix(model_utils.get_vit_config(model).hidden_size,
                                                args.rotate_mode, args.seed)
    Q1 = get_orthogonal_matrix(model_utils.get_vit_config(model).intermediate_size,
                                                args.rotate_mode)
    config = model_utils.get_vit_config(model)
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.get_model_typevit(model)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type) # write new mlp_input for online rotation here
        W = layer.mlp.fc2
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q1).to(device="cpu", dtype=dtype)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
        layer.layer_norm1.online_random_had = True
        layer.layer_norm1.had_K = Q
        layer.layer_norm1.K = 1
        layer.layer_norm1.fp32_had = args.fp32_had
        layer.layer_norm2.online_random_had = True
        layer.layer_norm2.had_K = Q
        layer.layer_norm2.K = 1
        layer.layer_norm2.fp32_had = args.fp32_had
    # add rotate projector layers, should be online
    if args.vit_mmoff:
        for proj in model_utils.get_projector(model, model_type):
            if type(proj) in [nn.Linear]:
                W = proj
            elif model_type == model_utils.LLAVA_NEXT_HF:
                W = proj.linear_1
            else:
                W = proj[0]
            apply_exact_had_to_linear(W, had_dim=-1, output=False)
        return Q1
    rotate_projectorsinput(model)
    return Q1


@torch.inference_mode()
def rotate_smovlmvitmmonlineR(model, args):
    Q = get_orthogonal_matrix(model.model.vision_model.config.hidden_size,
                                                args.rotate_mode, args.seed)
    Q1 = get_orthogonal_matrix(model.model.vision_model.config.intermediate_size,
                                                args.rotate_mode)
    config = model.model.vision_model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.get_model_typevit(model)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type) # write new mlp_input for online rotation here
        W = layer.mlp.fc2
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q1).to(device="cpu", dtype=dtype)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
        layer.layer_norm1.online_random_had = True
        layer.layer_norm1.had_K = Q
        layer.layer_norm1.K = 1
        layer.layer_norm1.fp32_had = args.fp32_had
        layer.layer_norm2.online_random_had = True
        layer.layer_norm2.had_K = Q
        layer.layer_norm2.K = 1
        layer.layer_norm2.fp32_had = args.fp32_had
    # add rotate projector layers, should be online
    if args.vit_mmoff:
        return Q1, None
    rotate_projectorsinput(model, Q)
    return Q1, Q


@torch.inference_mode()
def rotate_modelvitmmonlineR_mmR(model, args):
    Q = get_orthogonal_matrix(model_utils.get_vit_config(model).hidden_size,
                                                args.rotate_mode, args.seed)
    Q1 = get_orthogonal_matrix(model_utils.get_vit_config(model).intermediate_size,
                                                args.rotate_mode)
    Q2 = get_orthogonal_matrix(model.model.config.hidden_size,
                                                args.rotate_mode)
    config = model_utils.get_vit_config(model)
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.get_model_typevit(model)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type) # write new mlp_input for online rotation here
        W = layer.mlp.fc2
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q1).to(device="cpu", dtype=dtype)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
        layer.layer_norm1.online_random_had = True
        layer.layer_norm1.had_K = Q
        layer.layer_norm1.K = 1
        layer.layer_norm1.fp32_had = args.fp32_had
        layer.layer_norm2.online_random_had = True
        layer.layer_norm2.had_K = Q
        layer.layer_norm2.K = 1
        layer.layer_norm2.fp32_had = args.fp32_had
    # add rotate projector layers, should be online
    if args.vit_mmoff:
        # why add rotation if mm_off, to cancel off vit output
        for proj in model_utils.get_projector(model, model_type):
            if type(proj) in [nn.Linear]:
                W = proj
            elif model_type == model_utils.LLAVA_NEXT_HF:
                W = proj.linear_1
            else:
                W = proj[0]
            apply_exact_had_to_linear(W, had_dim=-1, output=False)
        return Q1, None
    rotate_projectorsinput_mmR(model, Q2)
    return Q1, Q2


@torch.inference_mode()
def rotate_modelvitmmonlinev3(model, args, rotations):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode, args.seed)
    R3, R4 = rotations['R3'].double(), rotations['R4'].double()
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    had_dim = R3.shape[0]


    model_type = model_utils.get_model_typevit(model)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type) # write new mlp_input for online rotation here
        if had_dim == model_dim * 4:
            W = layer.mlp.fc1
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(R3.T, W_).to(device="cpu", dtype=dtype)
            if W.bias is not None:
                b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
                W.bias.data = torch.matmul(R3.T, b).to(device="cpu", dtype=dtype)
            W = layer.mlp.fc2
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(W_, R4).to(device="cpu", dtype=dtype)
        else:
            W = layer.mlp.fc1
            dtype = W.weight.data.dtype
            temp = W.weight.t()
            init_shape = temp.shape
            temp = temp.reshape(-1, init_shape[0]//had_dim, had_dim).to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(temp, R3).reshape(init_shape).to(device="cpu", dtype=dtype).t()
            if W.bias is not None:
                init_shape = W.bias.shape
                temp = W.bias.data.reshape(had_dim, -1).to(device=utils.get_dev(), dtype=torch.float64)
                W.bias.data = torch.matmul(R3.T, temp).reshape(init_shape).to(device="cpu", dtype=dtype)
            W = layer.mlp.fc2
            dtype = W.weight.data.dtype
            init_shape = W.weight.shape
            temp = W.weight.data.reshape(-1, init_shape[0]//had_dim, had_dim).to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(temp, R4).reshape(init_shape).to(device="cpu", dtype=dtype)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
        layer.layer_norm1.online_random_had = True
        layer.layer_norm1.had_K = Q
        layer.layer_norm1.K = 1
        layer.layer_norm1.fp32_had = args.fp32_had
        layer.layer_norm2.online_random_had = True
        layer.layer_norm2.had_K = Q
        layer.layer_norm2.K = 1
        layer.layer_norm2.fp32_had = args.fp32_had
    # add rotate projector layers, should be online
    rotate_projectorsinput(model)


@torch.inference_mode()
def rotate_modelvitmmonlinev4(model, args, rotations):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode, args.seed)
    R3s, R4s, SQs= rotations['R3'], rotations['R4'], rotations['sq']
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    had_dim = R3s[0].shape[0]


    model_type = model_utils.get_model_typevit(model)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        R3, R4 = R3s[idx].double(), R4s[idx].double()
        if len(SQs) > 0:
            sq = SQs[idx].double()
        else:
            sq = None
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type) # write new mlp_input for online rotation here
        if had_dim == model_dim * 4:
            W = layer.mlp.fc1
            dtype = W.weight.data.dtype
            
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            if sq is not None:
                W_ *= sq[:, None].to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(R3.T, W_).to(device="cpu", dtype=dtype)
            if W.bias is not None:
                b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
                if sq is not None:
                    b *= sq.to(device=utils.get_dev(), dtype=torch.float64)
                W.bias.data = torch.matmul(R3.T, b).to(device="cpu", dtype=dtype)
            W = layer.mlp.fc2
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            if sq is not None:
                W_ /= sq.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(W_, R4).to(device="cpu", dtype=dtype)
        else:
            W = layer.mlp.fc1
            dtype = W.weight.data.dtype
            init_shape = W.weight.shape
            if sq is not None:
                W.weight.data = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
                W.weight.data *= sq[:, None].to(device=utils.get_dev(), dtype=torch.float64)
            temp = W.weight.data.reshape(init_shape[0]//had_dim, had_dim, -1).to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(R3.T, temp).reshape(init_shape).to(device="cpu", dtype=dtype)
            if W.bias is not None:
                init_shape = W.bias.shape
                if sq is not None:
                    W.bias.data=W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
                    W.bias.data *= sq.to(device=utils.get_dev(), dtype=torch.float64)
                temp = W.bias.data.reshape(had_dim, -1).to(device=utils.get_dev(), dtype=torch.float64)
                W.bias.data = torch.matmul(R3.T, temp).reshape(init_shape).to(device="cpu", dtype=dtype)
            W = layer.mlp.fc2
            dtype = W.weight.data.dtype
            init_shape = W.weight.shape
            if sq is not None:
                W.weight.data= W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
                W.weight.data /= sq.to(device=utils.get_dev(), dtype=torch.float64)
            temp = W.weight.data.reshape(-1, init_shape[0]//had_dim, had_dim).to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(temp, R4).reshape(init_shape).to(device="cpu", dtype=dtype)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
        layer.layer_norm1.online_random_had = True
        layer.layer_norm1.had_K = Q
        layer.layer_norm1.K = 1
        layer.layer_norm1.fp32_had = args.fp32_had
        layer.layer_norm2.online_random_had = True
        layer.layer_norm2.had_K = Q
        layer.layer_norm2.K = 1
        layer.layer_norm2.fp32_had = args.fp32_had
    # add rotate projector layers, should be online
    rotate_projectorsinput(model)


@torch.inference_mode()
def rotate_modelvitmmonlinev5(model, args, rotations):
    # v5, sq learn sperately for Win and Wout
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode, args.seed)
    R3s, R4s, SQs= rotations['R3'], rotations['R4'], rotations['sq']
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    had_dim = R3s[0].shape[0]


    model_type = model_utils.get_model_typevit(model)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        R3, R4 = R3s[idx].double(), R4s[idx].double()
        if len(SQs) > 0:
            sq, sq1 = SQs[idx][0].double(), SQs[idx][1].double()
        else:
            sq = None
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type) # write new mlp_input for online rotation here
        if had_dim == model_dim * 4:
            W = layer.mlp.fc1
            dtype = W.weight.data.dtype
            
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            if sq is not None:
                W_ /= sq1.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(R3.T, W_).to(device="cpu", dtype=dtype)
            if W.bias is not None:
                b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
                if sq is not None:
                    b /= sq1.to(device=utils.get_dev(), dtype=torch.float64)
                W.bias.data = torch.matmul(R3.T, b).to(device="cpu", dtype=dtype)
            W = layer.mlp.fc2
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            if sq is not None:
                W_ *= sq.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(W_, R4).to(device="cpu", dtype=dtype)
        else:
            W = layer.mlp.fc1
            dtype = W.weight.data.dtype
            init_shape = W.weight.shape
            if sq is not None:
                W.weight.data = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
                W.weight.data /= sq1.to(device=utils.get_dev(), dtype=torch.float64)
            temp = W.weight.data.reshape(init_shape[0]//had_dim, had_dim, -1).to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(R3.T, temp).reshape(init_shape).to(device="cpu", dtype=dtype)
            if W.bias is not None:
                init_shape = W.bias.shape
                if sq is not None:
                    W.bias.data=W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
                    W.bias.data /= sq1.to(device=utils.get_dev(), dtype=torch.float64)
                temp = W.bias.data.reshape(had_dim, -1).to(device=utils.get_dev(), dtype=torch.float64)
                W.bias.data = torch.matmul(R3.T, temp).reshape(init_shape).to(device="cpu", dtype=dtype)
            W = layer.mlp.fc2
            dtype = W.weight.data.dtype
            init_shape = W.weight.shape
            if sq is not None:
                W.weight.data= W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
                W.weight.data *= sq.to(device=utils.get_dev(), dtype=torch.float64)
            temp = W.weight.data.reshape(-1, init_shape[0]//had_dim, had_dim).to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(temp, R4).reshape(init_shape).to(device="cpu", dtype=dtype)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
        layer.layer_norm1.online_random_had = True
        layer.layer_norm1.had_K = Q
        layer.layer_norm1.K = 1
        layer.layer_norm1.fp32_had = args.fp32_had
        layer.layer_norm2.online_random_had = True
        layer.layer_norm2.had_K = Q
        layer.layer_norm2.K = 1
        layer.layer_norm2.fp32_had = args.fp32_had
    # add rotate projector layers, should be online
    rotate_projectorsinput(model)


@torch.inference_mode()
def rotate_modelvitonlinev1(model, args):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode, args.seed)
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.get_model_typevit(model)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
            apply_exact_had_to_linear(W, had_dim=-1, output=False)
        W = layer.mlp.fc1
        apply_exact_had_to_linear(W, had_dim=-1, output=False)
        # # rotate_mlp_output(layers[idx], Q, model_type) # no in vit do not need 
        W = layer.mlp.fc2
        apply_exact_had_to_linear(W, had_dim=-1, output=False)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)

@torch.inference_mode
def online_rotate(module, inp):
    x = torch.nn.functional.linear(inp[0], module.Q)
    return (x,) + inp[1:]

def register_online_rotation(module, Q:torch.Tensor):
    assert not hasattr(module, 'Q')
    module.register_buffer('Q', Q.T.to(module.weight.data))  # Note F.linear(x, A) performs x@A.T

    # We use forward_pre_hook because we capture the input using forward_hook, which could then capture the rotated input.
    # If we implement in the forward() the un-rotated original input will be captured.
    module.rotate_handle = module.register_forward_pre_hook(online_rotate)


class QKRotationWrapper(torch.nn.Module):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(head_dim), f'Only power of 2 head_dim is supported for K-cache Quantization!'
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs['k_groupsize'] in [-1, head_dim], f'Only token-wise/{head_dim}g quantization is supported for K-cache'
            self.k_bits = kwargs['k_bits']
            self.k_groupsize = kwargs['k_groupsize']
            self.k_sym = kwargs['k_sym']
            self.k_clip_ratio = kwargs['k_clip_ratio']
            self.k_quantizer.configure(bits=self.k_bits, groupsize=-1, #we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                                   sym=self.k_sym, clip_ratio=self.k_clip_ratio)

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = hadamard_transform(q.float(), scale=1/math.sqrt(q.shape[-1])).to(dtype)
        k = hadamard_transform(k.float(), scale=1/math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape
        

        if self.k_groupsize == -1: #token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, self.config.hidden_size)
            self.k_quantizer.find_params(token_wise_k)
            k = self.k_quantizer(token_wise_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
        else: #head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = self.k_quantizer(per_head_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)
        
        self.k_quantizer.free()
            
        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward. 
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    import monkeypatch
    import functools
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(module, "forward",
                                                                    function_name, functools.partial(QKRotationWrapper, *args, **kwargs))
    setattr(module, attr_name, wrapper)


class QKfp8RotationWrapper(torch.nn.Module):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(head_dim), f'Only power of 2 head_dim is supported for K-cache Quantization!'
        self.func = func
        self.k_quantizer = quant_utils.Actfp8Quantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs['k_groupsize'] in [-1, head_dim], f'Only token-wise/{head_dim}g quantization is supported for K-cache'
            self.k_bits = kwargs['k_bits']
            self.k_groupsize = kwargs['k_groupsize']
            self.k_sym = kwargs['k_sym']
            self.k_clip_ratio = kwargs['k_clip_ratio']
            self.k_quantizer.configure(bits=self.k_bits, groupsize=-1, #we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                                   sym=self.k_sym, clip_ratio=self.k_clip_ratio)

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = hadamard_transform(q.float(), scale=1/math.sqrt(q.shape[-1])).to(dtype)
        k = hadamard_transform(k.float(), scale=1/math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape
        

        if self.k_groupsize == -1: #token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, self.config.hidden_size)
            self.k_quantizer.find_params(token_wise_k)
            k = self.k_quantizer(token_wise_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
        else: #head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = self.k_quantizer(per_head_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)
        
        self.k_quantizer.free()
            
        return q, k

def addfp8_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward. 
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    import monkeypatch
    import functools
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(module, "forward",
                                                                    function_name, functools.partial(QKfp8RotationWrapper, *args, **kwargs))
    setattr(module, attr_name, wrapper)


def rotate_per_layer(model, dataloader, dev, args, tokenizer=None, image_processor=None):
    logging.info('-----Learn Q vit -----')
    dataloader, _ = dataloader
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.vision_tower.vision_tower.vision_model.encoder.layers
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.model.vision_tower.vision_tower.vision_model.parameters())).dtype
    R1 = None
    R2 = None
    if args.rot_dim:
        R3 = get_orthogonal_matrix(args.rot_dim,
                                    args.rotate_mode) # applied after fc1
        R4 = get_orthogonal_matrix(args.rot_dim,
                                    args.rotate_mode) # applied after fc1
    else:
        R3 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                    args.rotate_mode) # applied after fc1
        R4 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                args.rotate_mode) # applied before fc2
    ## have
    R1 = nn.Parameter(R1.to(torch.float32).to(utils.get_dev())) if R1 is not None else None
    R2 = nn.Parameter(R2.to(torch.float32).to(utils.get_dev())) if R2 is not None else None
    R3 = nn.Parameter(R3.to(torch.float32).to(utils.get_dev())) if R3 is not None else None
    R4 = nn.Parameter(R4.to(torch.float32).to(utils.get_dev())) if R4 is not None else None
    inps = torch.zeros(
        (args.nsamples, model.model.vision_tower.num_patches+1, model.model.vision_tower.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            input_ids, images = gptq_utils.message_to_prompt(batch, image_processor, model, tokenizer)
            from llava.constants import IMAGE_TOKEN_INDEX
            model.generate(input_ids, images=images,
                            do_sample=False,
                            temperature=0,
                            max_new_tokens=512,
                            use_cache=True,)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    outsfp16 = torch.zeros_like(inps)
    

    rots = {}
    for i in range(len(layers)):
        torch.set_grad_enabled(True)
        logging.info(f'\nLayer {i}:\n', flush=True, end=' ')
        layer = layers[i].to(dev)
        for param in layer.parameters():
            param.requires_grad = False
        outsfp16 = layer(inps, None, None)
        layer = model_utils.Layerwrapper(layer, R1=R1, R2=R2, R3=R3, R4=R4) # here add forward to fuse weight etc
        layer.init_rot()
        print("using rot")
        layer.init_quant(args)
        print("using quant")
        loss_fn = nn.MSELoss()
        trainable_parameters = [layer.R3, layer.R4]
        optimizer = SGDG(trainable_parameters, lr=args.rot_lr, stiefel=True)
        for epoch in range(args.rot_epochs):
            for j in range(args.nsamples):
                optimizer.zero_grad()
                out = layer(inps[j].unsqueeze(0), None, None)[0]
                loss = ((outsfp16[j]-out)**2).sum(-1).mean()
                loss.backward()
                optimizer.step()
                if j % 50 == 0:
                    logging.info(f"Epoch {epoch}, Iter {j}, Loss: {loss.item()}")
        layer.reset_rot()
        layers[i] = layer.module.cpu() # here delete the wrapper
        del layer
        torch.cuda.empty_cache()
        logging.info('now testing only 1 layer results')
        break
        inps, outsfp16 = outsfp16, inps
    rots = {'R1':R1, 'R2':R2, 'R3':R3, 'R4': R4}
    return rots


def rotate_per_layermlp(model, dataloader, dev, args, tokenizer=None, image_processor=None):
    logging.info('-----Learn Q vit mlp-----')
    dataloader, _ = dataloader
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.vision_tower.vision_tower.vision_model.encoder.layers
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.model.vision_tower.vision_tower.vision_model.parameters())).dtype
    def init_R():
        R1 = None
        R2 = None
        if args.rot_dim:
            R3 = get_orthogonal_matrix(args.rot_dim,
                                        args.rotate_mode) # applied after fc1
            R4 = get_orthogonal_matrix(args.rot_dim,
                                        args.rotate_mode) # applied after fc1
        else:
            R3 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                        args.rotate_mode) # applied after fc1
            R4 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                    args.rotate_mode) # applied before fc2
        R1 = nn.Parameter(R1.to(torch.float32).to(utils.get_dev())) if R1 is not None else None
        R2 = nn.Parameter(R2.to(torch.float32).to(utils.get_dev())) if R2 is not None else None
        R3 = nn.Parameter(R3.to(torch.float32).to(utils.get_dev())) if R3 is not None else None
        R4 = nn.Parameter(R4.to(torch.float32).to(utils.get_dev())) if R4 is not None else None
        return R1, R2, R3, R4
    def init_R_I():
        R1 = None
        R2 = None
        if args.rot_dim:
            R3 = torch.eye(args.rot_dim)
            R4 = torch.eye(args.rot_dim)
        else:
            R3 = torch.eye(model.model.vision_tower.config.intermediate_size)
            R4 = torch.eye(model.model.vision_tower.config.intermediate_size)
        R1 = nn.Parameter(R1.to(torch.float32).to(utils.get_dev())) if R1 is not None else None
        R2 = nn.Parameter(R2.to(torch.float32).to(utils.get_dev())) if R2 is not None else None
        R3 = nn.Parameter(R3.to(torch.float32).to(utils.get_dev())) if R3 is not None else None
        R4 = nn.Parameter(R4.to(torch.float32).to(utils.get_dev())) if R4 is not None else None
        return R1, R2, R3, R4
    def init_sq(sq = False, weight=None):
        if sq:
            return (nn.Parameter(torch.tensor(1).to(torch.float32).to(utils.get_dev())), nn.Parameter(torch.tensor(1).to(torch.float32).to(utils.get_dev())))
        else:
            return None
    inps = torch.zeros(
        (args.nsamples, model.model.vision_tower.num_patches+1, model.model.vision_tower.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0}
    cacheresidual = {'i': 0}
    residual = torch.zeros_like(inps)
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError
    class residualCatcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            residual[[cacheresidual['i']]] = inp
            cacheresidual['i'] += 1
            return self.module(inp) # pass through layernorm
    layers[0].mlp = Catcher(layers[0].mlp)
    layers[0].layer_norm2 = residualCatcher(layers[0].layer_norm2)
    for batch in dataloader:
        try:
            input_ids, images = gptq_utils.message_to_prompt(batch, image_processor, model, tokenizer)
            from llava.constants import IMAGE_TOKEN_INDEX
            model.generate(input_ids, images=images,
                            do_sample=False,
                            temperature=0,
                            max_new_tokens=512,
                            use_cache=True,)
        except ValueError:
            if cache['i'] == 1:
                print('valueerror')
            pass
    layers[0].mlp = layers[0].mlp.module
    layers[0].layer_norm2 = layers[0].layer_norm2.module
    layers[0] = layers[0].cpu()
    torch.cuda.empty_cache()
    outsfp16 = torch.zeros_like(inps)
    

    rots = {}
    rots = {'R1':[], 'R2':[], 'R3':[], 'R4': [], 'sq': []}
    for i in range(len(layers)):
        torch.set_grad_enabled(True)
        print(f'\nLayer {i}:', flush=True, end=' ')

        layer = layers[i].mlp.to(dev)
        for param in layer.parameters():
            param.requires_grad = False
        R1, R2, R3, R4 = init_R()
        sq = init_sq(args.sq, weight = layer)
        outsfp16 = layer(inps)
        layer = model_utils.mlpLayerwrapper(layer, R1=R1, R2=R2, R3=R3, R4=R4, sq=sq) # here add forward to fuse weight etc
        layer.init_rot()
        loss_fn = nn.MSELoss()
        trainable_parameters = []
        trainable_parameters.append({"params": layer.R3, "lr": args.rot_lr})
        trainable_parameters.append({"params": layer.R4, "lr": args.rot_lr})
        if sq is not None:
            trainable_parameters.append({"params": layer.sq, "lr": args.sq_lr})
            trainable_parameters.append({"params": layer.sq1, "lr": args.sq_lr})
        optimizer = torch.optim.SGD(trainable_parameters)
        loss_curve = []
        loss_curve_normabs = []
        for epoch in range(args.rot_epochs):
            for j in range(args.nsamples):
                optimizer.zero_grad()
                out = layer(inps[j])
                loss = loss_fn(outsfp16[j], out) 
                if torch.isnan(loss):
                    continue
                loss_curve.append(loss.item())
                loss /= outsfp16[j].abs().mean()
                loss.backward()
                optimizer.step()
                loss_curve_normabs.append(loss.item())
                if j % 50 == 0 and epoch % 5 ==0:
                    logging.info(f"Epoch {epoch}, Iter {j}, Loss: {loss.item()}")
        logging.info(f"Epoch {epoch}, Iter {j}, Loss: {loss.item()}")
        def save_plot(loss_curve, name=None):
            import matplotlib.pyplot as plt
            import os
            save_path = args.save_path + '/save_loss/'
            os.makedirs(save_path, exist_ok=True)
            plt.figure(figsize=(10, 5))
            iters = [_ for _ in range(args.nsamples * args.rot_epochs)]
            plt.plot(iters, loss_curve, label = 'loss', linewidth=2)
            epochs = [_ for _ in range(args.rot_epochs)]
            epochs_iter = [_* args.nsamples for _ in range(args.rot_epochs)]
            plt.xlabel("Epochs")
            plt.xticks(epochs_iter, epochs)
            plt.ylabel("loss")
            plt.title(f"Loss curve-layer{i}")
            plt.savefig(f'{save_path}lr_{args.rot_lr}_sgd_r3_layer_{i}_{name}.png')
            plt.close()
        save_plot(loss_curve_normabs,'normabs')
        layer.reset_rot()
        layers[i].mlp = layer.module.cpu() # here delete the wrapper
        del layer
        torch.cuda.empty_cache()
        if i < 23:
            class Catchers(nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                def forward(self, inp, *args, **kwargs):
                    inps[:] = inp
                    cache['i'] += 1
                    raise ValueError
            class residualCatchers(nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                def forward(self, inp, *args, **kwargs):
                    residual[:] = inp
                    cacheresidual['i'] += 1
                    return self.module(inp) # pass through layernorm
            torch.set_grad_enabled(False)
            layer = layers[i+1].to(dev)
            cache = {'i': 0}
            cacheresidual = {'i': 0}
            layer.mlp = Catchers(layer.mlp)
            layer.layer_norm2 = residualCatchers(layer.layer_norm2)
            try:
                layer(outsfp16 + residual, None, None)
            except ValueError:
                print('valueerror')
                pass
            layer.mlp = layer.mlp.module
            layer.layer_norm2 = layer.layer_norm2.module
            layers[i+1] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
        rots['R3'].append(R3.clone().detach())
        rots['R4'].append(R4.clone().detach())
        if sq is not None:
            rots['sq'].append(sq)

    return rots


def rotate_per_layermlponline(model, dataloader, dev, args, tokenizer=None, image_processor=None):
    logging.info('-----Learn Q vit mlp-----')
    dataloader, _ = dataloader
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.vision_tower.vision_tower.vision_model.encoder.layers
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.model.vision_tower.vision_tower.vision_model.parameters())).dtype
    def init_R():
        R1 = None
        R2 = None
        if args.rot_dim:
            R3 = get_orthogonal_matrix(args.rot_dim,
                                        args.rotate_mode) # applied after fc1
            R4 = get_orthogonal_matrix(args.rot_dim,
                                        args.rotate_mode) # applied after fc1
        else:
            R3 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                        args.rotate_mode) # applied after fc1
            R4 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                    args.rotate_mode) # applied before fc2
        R1 = nn.Parameter(R1.to(torch.float32).to(utils.get_dev())) if R1 is not None else None
        R2 = nn.Parameter(R2.to(torch.float32).to(utils.get_dev())) if R2 is not None else None
        R3 = nn.Parameter(R3.to(torch.float32).to(utils.get_dev())) if R3 is not None else None
        R4 = nn.Parameter(R4.to(torch.float32).to(utils.get_dev())) if R4 is not None else None
        return R1, R2, R3, R4
    def init_sq(sq = False, weight=None):
        if sq:
            fc1_max = weight.fc1.weight.data.max(-1)[0]
            fc2_max = weight.fc2.weight.data.max(0)[0]
            ratio_max = fc2_max / fc1_max
            return nn.Parameter(ratio_max.to(torch.float32).to(utils.get_dev()))
        else:
            return None
    inps = torch.zeros(
        (args.nsamples, model.model.vision_tower.num_patches+1, model.model.vision_tower.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0}
    cacheresidual = {'i': 0}
    residual = torch.zeros_like(inps)
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError
    class residualCatcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            residual[[cacheresidual['i']]] = inp
            cacheresidual['i'] += 1
            return self.module(inp) # pass through layernorm
    class Catchers(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inps = inp
            cache['i'] += 1
            raise ValueError
    class residualCatchers(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            residual = inp
            cacheresidual['i'] += 1
            return self.module(inp) # pass through layernorm
    layers[0].mlp = Catcher(layers[0].mlp)
    layers[0] = residualCatchers(layers[0].layer_norm2)
    for batch in dataloader:
        try:
            input_ids, images = gptq_utils.message_to_prompt(batch, image_processor, model, tokenizer)
            from llava.constants import IMAGE_TOKEN_INDEX
            model.generate(input_ids, images=images,
                            do_sample=False,
                            temperature=0,
                            max_new_tokens=512,
                            use_cache=True,)
        except ValueError:
            pass
    layers[0].mlp = layers[0].mlp.module
    layers[0].layer_norm2 = layers[0].layer_norm2.module
    layers[0] = layers[0].cpu()
    torch.cuda.empty_cache()
    outsfp16 = torch.zeros_like(inps)
    
    
    rots = {}
    rots = {'R1':[], 'R2':[], 'R3':[], 'R4': [], 'sq': []}
    for i in range(len(layers)):
        torch.set_grad_enabled(True)
        print(f'\nLayer {i}:', flush=True, end=' ')

        layer = layers[i].mlp.to(dev)
        for param in layer.parameters():
            param.requires_grad = False
        R1, R2, R3, R4 = init_R()
        sq = init_sq(args.sq, weight = layer)
        outsfp16 = layer(inps)
        layer = model_utils.mlpLayerwrapper(layer, R1=R1, R2=R2, R3=R3, R4=R4, sq=sq) # here add forward to fuse weight etc
        layer.init_onlinerot(args)
        layer.init_quant(args)
        loss_fn = nn.MSELoss()
        
        trainable_parameters = [layer.R4]
        if sq is not None:
            trainable_parameters += [layer.sq]
        optimizer = torch.optim.SGD(trainable_parameters, lr=args.rot_lr)
        loss_curve = []
        loss_curve_normabs = []
        for epoch in range(args.rot_epochs):
            for j in range(args.nsamples):
                optimizer.zero_grad()
                out = layer(inps[j])
                loss = loss_fn(outsfp16[j], out) 
                if torch.isnan(loss):
                    continue
                loss_curve.append(loss.item())
                loss /= outsfp16[j].abs().mean()
                loss.backward()
                optimizer.step()
                loss_curve_normabs.append(loss.item())
                if j % 50 == 0 and epoch % 5 ==0:
                    logging.info(f"Epoch {epoch}, Iter {j}, Loss: {loss.item()}")
        logging.info(f"Epoch {epoch}, Iter {j}, Loss: {loss.item()}")
        def save_plot(loss_curve, name=None):
            import matplotlib.pyplot as plt
            import os
            save_path = args.save_path + '/save_loss/'
            os.makedirs(save_path, exist_ok=True)
            plt.figure(figsize=(10, 5))
            iters = [_ for _ in range(args.nsamples * args.rot_epochs)]
            plt.plot(iters, loss_curve, label = 'loss', linewidth=2)
            epochs = [_ for _ in range(args.rot_epochs)]
            epochs_iter = [_* args.nsamples for _ in range(args.rot_epochs)]
            plt.xlabel("Epochs")
            plt.xticks(epochs_iter, epochs)
            plt.ylabel("loss")
            plt.title(f"Loss curve-layer{i}")
            plt.savefig(f'{save_path}lr_{args.rot_lr}_sgd_r3_layer_{i}_{name}.png')
            plt.close()
        save_plot(loss_curve_normabs,'normabs')
        layer.reset_onlinerot()
        layers[i].mlp = layer.module.cpu() # here delete the wrapper
        del layer
        torch.cuda.empty_cache()
        if i < 23:
            torch.set_grad_enabled(False)
            layer = layers[i+1].to(dev)
            cache = {'i': 0}
            cacheresidual = {'i': 0}
            layer.mlp = Catchers(layer.mlp)
            layer.layer_norm2 = residualCatcher(layer.layer_norm2)
            try:
                layer(outsfp16 + residual, None, None)
            except ValueError:
                pass
            layer.mlp = layer.mlp.module
            layer.layer_norm2 = layer.layer_norm2.module
            layers[i+1] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
        rots['R3'].append(R3.clone().detach())
        rots['R4'].append(R4.clone().detach())
        if sq is not None:
            rots['sq'].append(sq.clone().detach())

    return rots


def rotate_projectorsinputllava_fusedsvd(model, Q: torch.Tensor) -> None:
    model_type = model_utils.model_type_extractor(model)
    for proj in model_utils.get_projector(model, model_type):
        W = proj[0].BLinear
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        W_ = proj[2].BLinear
        apply_exact_had_to_linear(W_, had_dim=-1, output=False)

def rotate_attention_inputssvd(layer, Q, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj.BLinear, layer.self_attn.k_proj.BLinear, layer.self_attn.v_proj.BLinear]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_attention_outputsvd(layer, Q, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.self_attn.o_proj.ALinear
    elif model_type == model_utils.OPT_MODEL or model_type == model_utils.LLAVA_MODEL:
        W = layer.self_attn.out_proj.ALinear
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_mlp_inputsvd(layer, Q, model_type):
    # Rotate the MLP input weights.
    if model_type == model_utils.LLAMA_MODEL:
        mlp_inputs = [layer.mlp.up_proj.BLinear, layer.mlp.gate_proj.BLinear]
    elif model_type == model_utils.OPT_MODEL:
        mlp_inputs = [layer.fc1.BLinear]
    elif model_type == model_utils.LLAVA_MODEL:
        mlp_inputs = [layer.mlp.fc1.BLinear]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_outputvitsvd(layer, Q, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj.ALinear
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2.ALinear
    elif model_type == model_utils.LLAVA_MODEL:
        W = layer.mlp.fc2.ALinear
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_ov_projvitsvd(layer, model_type, head_num, head_dim):
    v_proj = layer.self_attn.v_proj.ALinear
    if model_type == model_utils.LLAMA_MODEL:
        o_proj = layer.self_attn.o_proj.BLinear
    elif model_type == model_utils.OPT_MODEL or model_type == model_utils.LLAVA_MODEL:
        o_proj = layer.self_attn.out_proj.BLinear
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)

def rotate_ov_projsvd(layer, model_type, head_num, head_dim):
    v_proj = layer.self_attn.v_proj.ALinear
    if model_type == model_utils.LLAMA_MODEL:
        o_proj = layer.self_attn.o_proj.BLinear
    elif model_type == model_utils.OPT_MODEL or model_type == model_utils.LLAVA_MODEL:
        o_proj = layer.self_attn.out_proj.BLinear
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)


def rotate_ov_projsvdkv(layer, model_type, head_num, head_dim, svd_modules=None, is_per_head_svd=False):
    v_proj = layer.self_attn.v_proj.ALinear
    if model_type in [model_utils.LLAMA_MODEL, model_utils.SMOVLM_MODEL, model_utils.LLAVA_NEXT_HF, model_utils.MOLMO_MODEL]:
        if svd_modules in ['attn', 'all']:
            o_proj = layer.self_attn.o_proj.BLinear
        else:
            o_proj = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL or model_type == model_utils.LLAVA_MODEL:
        o_proj = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    if is_per_head_svd:
        hadamard_utils.apply_exact_had_to_perhead_linear(v_proj, had_dim=head_dim, output=True)
    else:
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)

def had_transform_rank(model):
    """
    Apply Hadamard transform to all eligible SVDLinear layers in the model.
    
    Args:
        model: Model to which Hadamard transform should be applied.
    """
    from . import svd_utils
    
    print("Starting Hadamard transform application for the model...")
    
    # Find all SVDLinear layers in the model
    svd_modules = []
    for name, module in model.named_modules():
        if isinstance(module, svd_utils.SVDLinear):
            svd_modules.append((name, module))
    
    if not svd_modules:
        print("No SVDLinear layers found in the model; cannot apply Hadamard transform")
        return
    
    print(f"Found {len(svd_modules)} SVDLinear layers")
    
    # Apply Hadamard transform to each SVDLinear layer
    applied_count = 0
    for name, module in svd_modules:
        if hasattr(module, 'had_K') and module.had_K is not None:
            try:
                module.apply_had_rank()
                applied_count += 1
            except Exception as e:
                print(f"Error applying Hadamard transform to layer {name}: {e}")
    
    # Print statistics for Hadamard transform application
    print(f"Successfully applied Hadamard transform to {applied_count} SVDLinear layers")
