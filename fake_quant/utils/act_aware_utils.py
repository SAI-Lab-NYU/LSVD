import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from . import (
    utils,
    gptq_utils,
    data_utils,
    model_utils,
    quant_utils,
    debug_utils,
    qwen_vlm_utils,
    molmo_utils,
    reasoning_calib_utils,
    calib_cache,
)
import logging

def _is_qwen_model(args, model, tokenizer):
    return qwen_vlm_utils.is_qwen_vlm_family(args=args, model=model, tokenizer=tokenizer)


def _is_lm_layer_module_name(name):
    return (
        "model.layers" in name
        or "language_model.layers" in name
        or "text_model.layers" in name
        or "transformer.blocks" in name
    )


def _build_qwen_inputs(message, image_processor, device, args=None, model=None, tokenizer=None):
    return qwen_vlm_utils.build_qwen_vlm_eval_inputs(
        message,
        image_processor,
        device,
        args=args,
        model=model,
        tokenizer=tokenizer,
    )

def _build_act_calib_quantizer(args, method):
    # Apply fake-quantized stats regardless of calibration method when enabled.
    method_l = str(method).lower()
    enable = (
        bool(getattr(args, "act_calib_fake_quant", False))
        or "fakequant" in method_l
        or "fq" in method_l
    )
    if not enable:
        return None

    a_bits = int(getattr(args, "a_bits_calib", 16))
    if a_bits >= 16:
        logging.info("Activation fake-quant stats enabled but a_bits=%d, skipping fake quant.", a_bits)
        return None

    quantizer = quant_utils.ActQuantizer()
    quantizer.configure(
        bits=a_bits,
        groupsize=int(getattr(args, "a_groupsize", -1)),
        sym=not bool(getattr(args, "a_asym", False)),
        clip_ratio=float(getattr(args, "a_clip_ratio", 1.0)),
    )
    logging.info(
        "Activation fake-quant stats enabled: bits=%d, groupsize=%d, sym=%s, clip_ratio=%.4f",
        a_bits,
        int(getattr(args, "a_groupsize", -1)),
        str(not bool(getattr(args, "a_asym", False))),
        float(getattr(args, "a_clip_ratio", 1.0)),
    )
    return quantizer


def _get_calib_stats_input(x, quantizer):
    # Keep model forward untouched; only fake-quantize a detached copy for statistics.
    if quantizer is None:
        return x.detach()
    x_stats = x.detach()
    try:
        quantizer.find_params(x_stats)
        x_stats = quantizer(x_stats)
        quantizer.free()
        return x_stats
    except Exception as e:
        logging.warning("Activation fake-quant stats failed once, fallback to fp activations: %s", e)
        try:
            quantizer.free()
        except Exception:
            pass
        return x.detach()

def _act_calib_cache_tag(args, method):
    method_l = str(method).lower()
    enable = (
        bool(getattr(args, "act_calib_fake_quant", False))
        or "fakequant" in method_l
        or "fq" in method_l
    )
    if not enable:
        return ""

    a_bits = int(getattr(args, "a_bits_calib", 16))
    a_groupsize = int(getattr(args, "a_groupsize", -1))
    a_asym = bool(getattr(args, "a_asym", False))
    a_clip_ratio = float(getattr(args, "a_clip_ratio", 1.0))
    clip_tag = str(a_clip_ratio).replace(".", "p")
    sym_tag = "asym" if a_asym else "sym"
    return f"_afq_ab{a_bits}_ag{a_groupsize}_{sym_tag}_ac{clip_tag}"


@torch.no_grad()
def calib_input_distribution(model, dataloader, tokenizer, image_processor, args, method, use_cache=True, cache_file=None):
    if args.a_bits_calib is None:
        args.a_bits_calib = args.a_bits
        
    model_id = model.config._name_or_path
    
    # Use the passed method parameter, if None try to get from args
    if method is None and hasattr(args, "calib_method"):
        method = args.calib_method
    if args.cache_file is None:
        cache_dir  = args.act_cache_dir + "/cache"
        if args.cache_in_log:
            cache_dir = args.act_cache_dir + "/cache" # just use cache_dir for all act aware to share
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = args.cache_file
        
    # Cache should reflect model state at SVD calibration time, not only args.rotate.
    # `pre_svd_rotate=True` means weights were rotated before SVD calibration.
    if bool(getattr(args, "pre_svd_rotate", False)):
        rotate_info = "presvdrot"
    else:
        rotate_info = "presvdnorot"
    if args.misalign_text_image:
        method += "_misalign"
    method_for_cache = (
        method
        + _act_calib_cache_tag(args, method)
        + reasoning_calib_utils.cache_tag(args, model=model, tokenizer=tokenizer, image_processor=image_processor)
    )
    if args.cache_file is not None:
        cache_file = os.path.join(cache_dir, f"{model_id.replace('/','_')}_{method_for_cache}_{rotate_info}_args{args.svd_modules}.pt")
    else:
        cache_file = os.path.join(cache_dir, f"{model_id.replace('/','_')}_{args.nsamples}_{args.seed}_calib_input_distribution_{method_for_cache}_{rotate_info}_args{args.svd_modules}.pt")
    cache_file = calib_cache.released_file(args, "act_input") or cache_file
    
    
    if os.path.exists(cache_file) and use_cache:
        logging.info(f"Loading calibration cache from {cache_file}...")
        all_scaling_diag_matrix = torch.load(cache_file, map_location="cpu")
        #### todo: add cholesky load
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'lm_head' not in name and _is_lm_layer_module_name(name):
                try:
                    if "cholesky" in method:
                        module.scaling_diag_matrixS = all_scaling_diag_matrix[name].float().to(
                            module.weight.device
                        ) if all_scaling_diag_matrix[name] is not None else None
                    else:
                        module.scaling_diag_matrix = all_scaling_diag_matrix[name].float().to(
                            module.weight.device
                        ) if all_scaling_diag_matrix[name] is not None else None
                except:
                    # for now pass the already svd modules loading
                    if "cholesky" in method:
                        module.scaling_diag_matrixS = None
                    else:
                        module.scaling_diag_matrix = None
                    pass
        logging.info("Successfully loaded calibration cache!")
        if getattr(args, "act_calib_debug", False) and "cholesky" in str(method).lower():
            logging.info(
                "act_calib_debug: cache hit — full input_covariance/Cholesky/inverse tensors were not recomputed; "
                "run with --no-use_cache or use_cache=False for online calib debug dump."
            )
        return
    
    logging.info("Starting online calibration...")
    logging.info(cache_file)
    model.eval()
    device = utils.get_dev()
    model = model.to(device)
    reasoning_batches = None
    if reasoning_calib_utils.enabled(args):
        reasoning_batches = reasoning_calib_utils.materialize_reasoning_batches(
            model=model,
            dataloader=dataloader,
            tokenizer=tokenizer,
            image_processor=image_processor,
            args=args,
            device=device,
        )
    # set hook for every Linear layers
    debug_hook_stats = bool(getattr(args, "debug", False)) or os.environ.get("ACT_AWARE_DEBUG", "0") == "1"
    hook_name_by_id = {}
    hook_calls = {}
    stats_quantizer = _build_act_calib_quantizer(args, method)

    def hook(module, input, output):
        module_id = id(module)
        if module_id in hook_name_by_id:
            hook_calls[module_id] = hook_calls.get(module_id, 0) + 1
        stats_inp = _get_calib_stats_input(input[0], stats_quantizer)
        lead_tokens = int(stats_inp.numel() // max(int(stats_inp.shape[-1]), 1))
        module._debug_sample_count = int(getattr(module, "_debug_sample_count", 0) + int(stats_inp.shape[0] if stats_inp.dim() > 1 else 1))
        module._debug_token_count = int(getattr(module, "_debug_token_count", 0) + lead_tokens)
        if "abs_mean" in method:
            abs_mean = stats_inp.abs().mean(dim=-2).detach().view(-1)
            module.scaling_diag_matrix += abs_mean.float()
        elif "abs_max" in method:
            abs_max = stats_inp.abs().amax(dim=-2).detach().view(-1)
            module.scaling_diag_matrix = torch.where(
                abs_max > module.scaling_diag_matrix,
                abs_max,
                module.scaling_diag_matrix,
            ).float()
        elif "cholesky" in method:
            inp = stats_inp.float()
            if inp.dim() == 2:   # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            if isinstance(module.raw_scaling_diag_matrix, torch.Tensor):
                module.raw_scaling_diag_matrix += adds_sum
            else:
                module.raw_scaling_diag_matrix = adds_sum
            del inp, adds, adds_sum
            torch.cuda.empty_cache()
    if "cholesky" in method:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and _is_lm_layer_module_name(name): # for now we only do SVD in LM
                if 'gate_proj' in name:
                    continue # qkv only store in k_proj, SwiGLU only store in up_proj, constantly skip
                elif 'o_proj' in name and args.svd_modules in ['qkv','kv','gaup','mlp','down']:
                    continue # only skip o_proj when svd_modules is qkv, kv, gaup, mlp, down
                elif 'mlp' in name and args.svd_modules in ['attn', 'qkv', 'kv', 'o']:
                    continue # only skip mlp when svd_modules is not all
                elif 'q_proj' in name and args.svd_modules in ['kv']:
                    continue # only skip q_proj when svd_modules is kv
                module.raw_scaling_diag_matrix = 0
                module_id = id(module)
                hook_name_by_id[module_id] = name
                hook_calls[module_id] = 0
                module.register_forward_hook(hook)
    else:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and _is_lm_layer_module_name(name): # for now we only do SVD in LM
                if 'gate_proj' in name:
                    continue # qkv only store in k_proj, SwiGLU only store in up_proj, constantly skip
                elif 'o_proj' in name and args.svd_modules in ['qkv','gaup','mlp', 'down']:
                    continue # only skip when svd_modules is qkv
                elif 'mlp' in name and args.svd_modules in ['attn', 'qkv', 'o']:
                    continue # only skip when svd_modules is not all
                elif 'q_proj' in name and args.svd_modules in ['kv']:
                    continue # only skip q_proj when svd_modules is kv
                module.scaling_diag_matrix = 0
                module_id = id(module)
                hook_name_by_id[module_id] = name
                hook_calls[module_id] = 0
                module.register_forward_hook(hook)

    # get activation distribution
    class Catcherout(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inp = self.module(inp)
            raise ValueError

    layers = model_utils.get_layers(model)
    num_layers = len(layers)
    layer = layers[num_layers-1]
    layer.mlp = Catcherout(layer.mlp)
    
    # Ensure the entire model is on CUDA
    model = model.to(utils.get_dev())

    total_seq_length = 0
    
    if reasoning_batches is not None:
        for reasoning_batch in reasoning_batches:
            try:
                total_seq_length += int(reasoning_batch.inputs["input_ids"].shape[1])
                reasoning_calib_utils.forward_reasoning_batch(model, reasoning_batch)
            except ValueError:
                pass
    else:
        for batch in dataloader:
            try:
                # Ensure model and input are on the same device
                device = utils.get_dev()  # Get the device where the model is currently located
                is_qwen_model = _is_qwen_model(args, model, tokenizer)
                
                # Use model to generate output
                if tokenizer is None and image_processor is None:
                    input_ids = batch[0].to(device)
                    model(input_ids)
                    
                elif is_qwen_model:
                    inputs = _build_qwen_inputs(
                        batch, image_processor, device, args=args, model=model, tokenizer=tokenizer
                    )
                    seq_length = inputs["input_ids"].shape[1]
                    total_seq_length += seq_length
                    model.generate(**inputs, max_new_tokens=1, use_cache=True)
                elif molmo_utils.is_molmo_model(args=args, model=model, tokenizer=tokenizer):
                    inputs = molmo_utils.build_molmo_eval_inputs(
                        batch,
                        image_processor,
                        device,
                        model=model,
                    )
                    total_seq_length += inputs["input_ids"].shape[1]
                    molmo_utils.forward_molmo_prefill(model, inputs)
                elif tokenizer is None or 'hf_v16' in str(tokenizer): # SmolVLM/llava v1.6 [HF version]
                    # Use message_to_prompt to process batch data
                    inputs, _ = gptq_utils.message_to_prompt(batch, image_processor, model, tokenizer)
                    inputs = inputs.to(device)

                    # Assume each batch contains only one sample, directly take its sequence length
                    seq_length = inputs['input_ids'].shape[1]
                    total_seq_length += seq_length

                    model.generate(**inputs,
                                    max_new_tokens=1,
                                    use_cache=True,)
                elif tokenizer == 'intern':
                    inputs = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer)
                    model.chat(
                        image_processor,
                        **inputs,
                    )
                else: # LLaVA v1.5
                    # Use message_to_prompt to process batch data
                    input_ids, images = gptq_utils.message_to_prompt(batch, image_processor, model, tokenizer)

                    # Ensure input_ids and images are on the correct device
                    input_ids = input_ids.to(device)
                    image_sizes = None
                    if images is not None:
                        images, image_sizes = images
                        images = images.to(device)

                    # Assume each batch contains only one sample, directly take its sequence length
                    seq_length = input_ids.shape[1]
                    total_seq_length += seq_length

                    from llava.constants import IMAGE_TOKEN_INDEX

                    model.generate(input_ids, images=images,
                                    image_sizes = image_sizes,
                                    do_sample=False,
                                    max_new_tokens=1,
                                    use_cache=True,)
            except ValueError:
                pass

    logging.info("Random Seed: %s", args.seed)
    logging.info("Total sequence length: %s", total_seq_length)
    if debug_hook_stats and hook_calls:
        triggered = [k for k, v in hook_calls.items() if v > 0]
        untriggered = [k for k, v in hook_calls.items() if v == 0]
        logging.warning(
            "[hook-debug] calib_input_distribution: total=%d, triggered=%d, untriggered=%d",
            len(hook_calls), len(triggered), len(untriggered),
        )
        if untriggered:
            untriggered_names = [hook_name_by_id[k] for k in untriggered]
            logging.warning("[hook-debug] untriggered sample: %s", ", ".join(untriggered_names[:20]))
        top_called = sorted(triggered, key=lambda k: hook_calls[k], reverse=True)[:20]
        if top_called:
            logging.warning(
                "[hook-debug] top called layers: %s",
                ", ".join([f"{hook_name_by_id[k]}({hook_calls[k]})" for k in top_called]),
            )
    
    layer.mlp = layer.mlp.module

    if "cholesky" in method:
        # remove and compute cholesky decomposition
        missed_hook_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and _is_lm_layer_module_name(name):
                if 'gate_proj' in name:
                    continue # qkv only store in k_proj, SwiGLU only store in up_proj, constantly skip
                elif 'o_proj' in name and args.svd_modules in ['qkv','gaup','mlp', 'down']:
                    continue # only skip when svd_modules is qkv
                elif 'mlp' in name and args.svd_modules in ['attn', 'qkv', 'o']:
                    continue # only skip when svd_modules is not all
                elif 'q_proj' in name and args.svd_modules in ['kv']:
                    continue # only skip q_proj when svd_modules is kv
                module._forward_hooks.clear()
                if isinstance(module.raw_scaling_diag_matrix, torch.Tensor):
                    module.raw_scaling_diag_matrix = module.raw_scaling_diag_matrix.cpu()
                else:
                    missed_hook_layers.append(name)
        if missed_hook_layers:
            logging.warning(
                "Cholesky calibration: %d linear layers did not collect stats (hook not triggered).",
                len(missed_hook_layers),
            )
            logging.warning("Untriggered layers: %s", ", ".join(missed_hook_layers))
        layers = model_utils.get_layers(model) # skip vision for now
        logging.info("Start Cholesky Decomposition...")
        act_calib_debug_payload = {}
        if getattr(args, "act_calib_debug", False):
            logging.info(
                "act_calib_debug: collecting per-layer input covariance, Cholesky L, and inverse (large tensors)."
            )
        for i in tqdm(range(len(layers))):
            for name, module in layers[i].named_modules():
                if isinstance(module, nn.Linear):
                    if 'gate_proj' in name:
                        continue # qkv only store in k_proj, SwiGLU only store in up_proj, constantly skip
                    elif 'o_proj' in name and args.svd_modules in ['qkv','gaup','mlp', 'down']:
                        continue # only skip when svd_modules is qkv
                    elif 'mlp' in name and args.svd_modules in ['attn', 'qkv', 'o']:
                        continue # only skip when svd_modules is not all
                    elif 'q_proj' in name and args.svd_modules in ['kv']:
                        continue # only skip q_proj when svd_modules is kv
                    if not isinstance(module.raw_scaling_diag_matrix, torch.Tensor):
                        continue
                    raw_scaling_diag_matrix = module.raw_scaling_diag_matrix.double().to(utils.get_dev())
                    diag_shift = torch.zeros(
                        raw_scaling_diag_matrix.shape[:-2] if raw_scaling_diag_matrix.ndim > 2 else (),
                        device=raw_scaling_diag_matrix.device,
                        dtype=raw_scaling_diag_matrix.dtype,
                    )
                    try:
                        scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix) # remain double precision
                    except Exception as e:
                        print("Warning: eigen scaling_diag_matrix is not positive!")
                        eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                        diag_shift = (- eigenvalues[..., 0] + 1e-6).clamp_min(0.0)
                        raw_scaling_diag_matrix += diag_shift[..., None, None] * torch.eye(raw_scaling_diag_matrix.shape[-1]).to(utils.get_dev())
                        scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix) # remain double precision
                        eigenvalues = None
                        del eigenvalues
                    if getattr(args, "act_calib_debug", False):
                        full_key = f"model.layers.{i}.{name}"
                        act_calib_debug_payload[full_key] = debug_utils.act_calib_debug_pack_with_summary(
                            raw_cpu_double=raw_scaling_diag_matrix.detach().cpu(),
                            chol_cpu=scaling_diag_matrix.detach().cpu(),
                            normalization_count=max(getattr(module, "_debug_token_count", 0), 1),
                            sample_count=getattr(module, "_debug_sample_count", 0),
                            token_count=getattr(module, "_debug_token_count", 0),
                            hook_count=hook_calls.get(id(module), 0),
                            diag_shift_cpu=diag_shift.detach().cpu(),
                        )
                    if 'k_proj' in name: # only valid for k_proj in original svd_modules ==qkv 
                        layers[i].self_attn.k_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                    elif 'o_proj' in name:
                        layers[i].self_attn.o_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                    elif 'up_proj' in name:
                        layers[i].mlp.up_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                    elif 'down_proj' in name:
                        layers[i].mlp.down_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                    scaling_diag_matrix = raw_scaling_diag_matrix = module.raw_scaling_diag_matrix = None
                    del scaling_diag_matrix, raw_scaling_diag_matrix, module.raw_scaling_diag_matrix
                    if hasattr(module, "_debug_sample_count"):
                        del module._debug_sample_count
                    if hasattr(module, "_debug_token_count"):
                        del module._debug_token_count
                    torch.cuda.empty_cache()

        if getattr(args, "act_calib_debug", False) and act_calib_debug_payload:
            debug_dir = getattr(args, "act_calib_debug_dir", None) or getattr(args, "save_path", ".")
            os.makedirs(debug_dir, exist_ok=True)
            _sn = getattr(args, "save_name", "debug")
            dbg_path = os.path.join(debug_dir, f"act_calib_debug_calib_{_sn}.pt")
            torch.save(act_calib_debug_payload, dbg_path)
            n_bytes = os.path.getsize(dbg_path)
            logging.info(
                "act_calib_debug: saved calibration tensors for %d modules to %s (~%.2f MiB)",
                len(act_calib_debug_payload),
                dbg_path,
                n_bytes / (1024 * 1024),
            )

        all_scaling_diag_matrix = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and _is_lm_layer_module_name(name):
                try:
                    all_scaling_diag_matrix[name] = module.scaling_diag_matrixS
                except:
                    all_scaling_diag_matrix[name] = None
    else:
        # remove and save scaling_diag_matrix
        all_scaling_diag_matrix = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and _is_lm_layer_module_name(name):
                try:
                    module._forward_hooks.clear()
                    all_scaling_diag_matrix[name] = module.scaling_diag_matrix
                except:
                    module.scaling_diag_matrix = None
                    all_scaling_diag_matrix[name] = None
            
    logging.info(f"Saving calibration cache to {cache_file}...")
    torch.save(all_scaling_diag_matrix, cache_file)
    logging.info("Calibration cache saved successfully!")


@torch.no_grad()
def calib_input_distributionlowresources(model, tokenizer, image_processor, args, method, use_cache=True, cache_file=None, dataloader=None):
    
    if args.a_bits_calib is None:
        args.a_bits_calib = args.a_bits
    if reasoning_calib_utils.enabled(args):
        raise NotImplementedError(
            "Reasoning calibration requires full-forward activation calibration; "
            "call calib_input_distribution instead of calib_input_distributionlowresources."
        )
        
    if bool(getattr(args, "pre_svd_rotate", False)):
        rotate_info = "presvdrot"
    else:
        rotate_info = "presvdnorot"
    legacy_qwen_kproj_cholesky = _is_qwen_model(args, model, tokenizer) and "cholesky" in str(method).lower()
    if legacy_qwen_kproj_cholesky:
        logging.info(
            "Qwen model detected: using legacy low-resource cholesky replay (k_proj-only stats, shared to q/k/v)."
        )

    model_id = model.config._name_or_path
    # Use the passed method parameter, if None try to get from args
    if method is None and hasattr(args, "calib_method"):
        method = args.calib_method
    
    if args.cache_file is None:
        cache_dir  = args.act_cache_dir + "/cache"
        if args.cache_in_log:
            cache_dir = args.act_cache_dir + "/cache" # just use cache_dir for all act aware to share
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = args.cache_file
        
    # Keep cache naming consistent with full calibration path.
    # `pre_svd_rotate` reflects the model state when SVD calibration runs.
    method_for_cache = method + _act_calib_cache_tag(args, method)
    if args.cache_file is not None:
        cache_file = os.path.join(cache_dir, f"{model_id.replace('/','_')}_{method_for_cache}_{rotate_info}_args{args.svd_modules}.pt")
    else:
        cache_file = os.path.join(cache_dir, f"{model_id.replace('/','_')}_{args.nsamples}_{args.seed}_calib_input_distribution_{method_for_cache}_{rotate_info}_args{args.svd_modules}.pt")
    cache_file = calib_cache.released_file(args, "act_input") or cache_file
        

    if os.path.exists(cache_file) and use_cache:
        logging.info(f"Loading calibration cache from {cache_file}...")
        all_scaling_diag_matrix = torch.load(cache_file, map_location="cpu")
        #### todo: add cholesky load
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'lm_head' not in name and _is_lm_layer_module_name(name):
                module.scaling_diag_matrixS = all_scaling_diag_matrix[name].to(
                    module.weight.device
                ) if all_scaling_diag_matrix[name] is not None else None
        logging.info("Successfully loaded calibration cache!")
        if getattr(args, "act_calib_debug", False) and "cholesky" in str(method).lower():
            logging.info(
                "act_calib_debug: cache hit — full input_covariance/Cholesky/inverse tensors were not recomputed; "
                "run with use_cache=False for online calib debug dump."
            )
        return
    
    logging.info("Starting online calibration...")
    model.eval()
    # set hook for every Linear layers
    debug_hook_stats = bool(getattr(args, "debug", False)) or os.environ.get("ACT_AWARE_DEBUG", "0") == "1"
    hook_name_by_id = {}
    hook_calls = {}
    stats_quantizer = _build_act_calib_quantizer(args, method)

    def hook(module, input, output):
        module_id = id(module)
        if module_id in hook_name_by_id:
            hook_calls[module_id] = hook_calls.get(module_id, 0) + 1
        stats_inp = _get_calib_stats_input(input[0], stats_quantizer)
        lead_tokens = int(stats_inp.numel() // max(int(stats_inp.shape[-1]), 1))
        module._debug_sample_count = int(getattr(module, "_debug_sample_count", 0) + int(stats_inp.shape[0] if stats_inp.dim() > 1 else 1))
        module._debug_token_count = int(getattr(module, "_debug_token_count", 0) + lead_tokens)
        if "abs_mean" in method:
            abs_mean = stats_inp.abs().mean(dim=-2).detach().view(-1)
            module.scaling_diag_matrix += abs_mean
        elif "abs_max" in method:
            abs_max = stats_inp.abs().amax(dim=-2).detach().view(-1)
            module.scaling_diag_matrix = torch.where(
                abs_max > module.scaling_diag_matrix,
                abs_max,
                module.scaling_diag_matrix,
            )
        elif "cholesky" in method:
            inp = stats_inp.float()
            if inp.dim() == 2:   # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            if isinstance(module.raw_scaling_diag_matrix, torch.Tensor):
                module.raw_scaling_diag_matrix += adds_sum
            else:
                module.raw_scaling_diag_matrix = adds_sum
            del inp, adds, adds_sum
            torch.cuda.empty_cache()

    # get activation distribution
    ######## for low resources, we just profile the first layer input, then do SVDLLM layer by layer
    ##### first layer input catcher
    if dataloader is None:
        # Load calibration dataset
        logging.info(f"Loading calibration dataset: {args.cal_dataset}")
        calib_loader = data_utils.get_loaders(
            args.cal_dataset, 
            nsamples=args.nsamples,
            seed=args.seed, 
            model=args.model,
            seqlen=model.seqlen, 
            eval_mode=False
        )
        dataloader, _  = calib_loader
    inps = []
    cache = {'i': 0, 'attention_mask': [], 'position_embeddings':[]}

    class _PrefillStop(Exception):
        """Internal control-flow signal to stop prefill after first-layer capture."""
        pass

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps.append(inp[0])
            cache['i'] += 1
            cache['attention_mask'].append(kwargs['attention_mask'])
            cache['position_embeddings'].append(kwargs['position_embeddings'])
            raise _PrefillStop
    layers = model_utils.get_layers(model)
    layers[0] = Catcher(layers[0])
    # Ensure the entire model is on CUDA
    model = model.to(utils.get_dev())
    for batch in dataloader:
            try:
                # Ensure model and input are on the same device
                device = utils.get_dev()  # Get the device where the model is currently located
                is_qwen_model = _is_qwen_model(args, model, tokenizer)
                if tokenizer is None and image_processor is None:
                    input_ids = batch[0].to(device)
                    model(input_ids)
                elif is_qwen_model:
                    inputs = _build_qwen_inputs(
                        batch, image_processor, device, args=args, model=model, tokenizer=tokenizer
                    )
                    model.generate(**inputs,
                                    max_new_tokens=1,
                                    use_cache=True,)
                elif molmo_utils.is_molmo_model(args=args, model=model, tokenizer=tokenizer):
                    inputs = molmo_utils.build_molmo_eval_inputs(
                        batch,
                        image_processor,
                        device,
                        model=model,
                    )
                    molmo_utils.forward_molmo_prefill(model, inputs)
                elif tokenizer is None or 'hf_v16' in str(tokenizer): # SmolVLM/llava1.6
                    # Use message_to_prompt to process batch data
                    inputs, _ = gptq_utils.message_to_prompt(batch, image_processor, model, tokenizer)
                    inputs = inputs.to(device)
                    
                    
                    model.generate(**inputs,
                                    max_new_tokens=1,
                                    use_cache=True,)
                elif tokenizer == 'intern':
                    inputs = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer)
                    model.chat(
                        image_processor,
                        **inputs,
                    )
                else: # LLaVA v1.5
                    # Use message_to_prompt to process batch data
                    input_ids, images = gptq_utils.message_to_prompt(batch, image_processor, model, tokenizer)
                    
                    # Ensure input_ids and images are on the correct device
                    input_ids = input_ids.to(device)
                    image_sizes = None
                    if images is not None:
                        images, image_sizes = images
                        images = images.to(device)
                        
        
                    from llava.constants import IMAGE_TOKEN_INDEX
                
                    model.generate(input_ids, images=images,
                                   image_sizes = image_sizes,
                                    do_sample=False,
                                    max_new_tokens=1,
                                    use_cache=True,)
            except _PrefillStop:
                # Expected from Catcher: used to stop full forward once layer-0 input is captured.
                pass
            except Exception as e:
                logging.exception("Unexpected error during low-resource calibration prefill: %s", e)
                raise
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    # try:
    #     model.model.embed_tokens = model.model.embed_tokens.cpu()
    #     model.model.norm = model.model.norm.cpu()
    # except:
    #     model.model.text_model.embed_tokens = model.model.text_model.embed_tokens.cpu()
    #     model.model.text_model.norm = model.model.text_model.norm.cpu()
    torch.cuda.empty_cache()
    outs = [None] * len(inps)
    missed_hook_layers = []
    ##### layer by layer input catcher
    for i in range(len(layers)):
        layer = layers[i].to(utils.get_dev())
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                if legacy_qwen_kproj_cholesky and 'k_proj' not in name:
                    continue
                if 'gate_proj' in name:
                    continue # qkv only store in k_proj, SwiGLU only store in up_proj, constantly skip
                elif 'o_proj' in name and args.svd_modules in ['qkv','kv','gaup','mlp', 'down']:
                    continue # only skip when svd_modules is qkv
                elif 'mlp' in name and args.svd_modules in ['attn', 'qkv', 'kv', 'o']:
                    continue # only skip when svd_modules is not all
                elif 'q_proj' in name and args.svd_modules in ['kv']:
                    continue # only skip q_proj when svd_modules is kv
                module.raw_scaling_diag_matrix = 0
                module_id = id(module)
                full_name = f"model.layers.{i}.{name}"
                hook_name_by_id[module_id] = full_name
                hook_calls[module_id] = 0
                module.register_forward_hook(hook)

        for j in range(len(inps)):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=cache['attention_mask'][j], position_embeddings=cache['position_embeddings'][j])[0][0]

        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                if legacy_qwen_kproj_cholesky and 'k_proj' not in name:
                    continue
                if 'gate_proj' in name:
                    continue # qkv only store in k_proj, SwiGLU only store in up_proj, constantly skip
                elif 'o_proj' in name and args.svd_modules in ['qkv','kv','gaup','mlp', 'down']:
                    continue # only skip when svd_modules is qkv
                elif 'mlp' in name and args.svd_modules in ['attn', 'qkv', 'kv', 'o']:
                    continue # only skip when svd_modules is not all
                elif 'q_proj' in name and args.svd_modules in ['kv']:
                    continue # only skip q_proj when svd_modules is kv
                module._forward_hooks.clear()
                if isinstance(module.raw_scaling_diag_matrix, torch.Tensor):
                    module.raw_scaling_diag_matrix = module.raw_scaling_diag_matrix.cpu()
                else:
                    missed_hook_layers.append(f"model.layers.{i}.{name}")
        inps = outs
    if debug_hook_stats and hook_calls:
        triggered = [k for k, v in hook_calls.items() if v > 0]
        untriggered = [k for k, v in hook_calls.items() if v == 0]
        logging.warning(
            "[hook-debug] calib_input_distributionlowresources: total=%d, triggered=%d, untriggered=%d",
            len(hook_calls), len(triggered), len(untriggered),
        )
        if untriggered:
            untriggered_names = [hook_name_by_id[k] for k in untriggered]
            logging.warning("[hook-debug] untriggered sample: %s", ", ".join(untriggered_names[:20]))
        top_called = sorted(triggered, key=lambda k: hook_calls[k], reverse=True)[:20]
        if top_called:
            logging.warning(
                "[hook-debug] top called layers: %s",
                ", ".join([f"{hook_name_by_id[k]}({hook_calls[k]})" for k in top_called]),
            )
    if missed_hook_layers:
        logging.warning(
            "Low-resource cholesky calibration: %d linear layers did not collect stats (hook not triggered).",
            len(missed_hook_layers),
        )
        logging.warning("Untriggered layers: %s", ", ".join(missed_hook_layers))
    logging.info("Start Cholesky Decomposition...")
    act_calib_debug_payload_lr = {}
    if getattr(args, "act_calib_debug", False) and "cholesky" in method:
        logging.info(
            "act_calib_debug (low-resource): collecting per-layer input covariance, Cholesky L, and inverse (large tensors)."
        )
    for i in tqdm(range(len(layers))):
        for name, module in layers[i].named_modules():
            if isinstance(module, nn.Linear):
                if legacy_qwen_kproj_cholesky and 'k_proj' not in name:
                    continue
                if 'gate_proj' in name:
                    continue # qkv only store in k_proj, SwiGLU only store in up_proj, constantly skip
                elif 'o_proj' in name and args.svd_modules in ['qkv','kv','gaup','mlp', 'down']:
                    continue # only skip when svd_modules is qkv
                elif 'mlp' in name and args.svd_modules in ['attn', 'qkv', 'kv', 'o']:
                    continue # only skip when svd_modules is not all
                elif 'q_proj' in name and args.svd_modules in ['kv']:
                    continue # only skip q_proj when svd_modules is kv
                if not isinstance(module.raw_scaling_diag_matrix, torch.Tensor):
                    continue
                raw_scaling_diag_matrix = module.raw_scaling_diag_matrix.double().to(utils.get_dev())
                diag_shift = torch.zeros(
                    raw_scaling_diag_matrix.shape[:-2] if raw_scaling_diag_matrix.ndim > 2 else (),
                    device=raw_scaling_diag_matrix.device,
                    dtype=raw_scaling_diag_matrix.dtype,
                )
                try:
                    scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).double()
                except Exception as e:
                    print("Warning: eigen scaling_diag_matrix is not positive!")
                    eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                    diag_shift = (- eigenvalues[..., 0] + 1e-6).clamp_min(0.0)
                    raw_scaling_diag_matrix += diag_shift[..., None, None] * torch.eye(raw_scaling_diag_matrix.shape[-1]).to(utils.get_dev())
                    scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).double()
                    eigenvalues = None
                    del eigenvalues
                if getattr(args, "act_calib_debug", False) and "cholesky" in method:
                    full_key = f"model.layers.{i}.{name}"
                    act_calib_debug_payload_lr[full_key] = debug_utils.act_calib_debug_pack_with_summary(
                        raw_cpu_double=raw_scaling_diag_matrix.detach().cpu(),
                        chol_cpu=scaling_diag_matrix.detach().cpu(),
                        normalization_count=max(getattr(module, "_debug_token_count", 0), 1),
                        sample_count=getattr(module, "_debug_sample_count", 0),
                        token_count=getattr(module, "_debug_token_count", 0),
                        hook_count=hook_calls.get(id(module), 0),
                        diag_shift_cpu=diag_shift.detach().cpu(),
                    )
                if 'k_proj' in name: # only valid for k_proj in original svd_modules ==qkv 
                    if legacy_qwen_kproj_cholesky:
                        layers[i].self_attn.q_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                        layers[i].self_attn.k_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                        layers[i].self_attn.v_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                    else:
                        layers[i].self_attn.k_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                elif 'o_proj' in name:
                    layers[i].self_attn.o_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                elif 'up_proj' in name:
                    layers[i].mlp.up_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                elif 'down_proj' in name:
                    layers[i].mlp.down_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                scaling_diag_matrix = raw_scaling_diag_matrix = module.raw_scaling_diag_matrix = None
                del scaling_diag_matrix, raw_scaling_diag_matrix, module.raw_scaling_diag_matrix
                if hasattr(module, "_debug_sample_count"):
                    del module._debug_sample_count
                if hasattr(module, "_debug_token_count"):
                    del module._debug_token_count
                torch.cuda.empty_cache()
    if getattr(args, "act_calib_debug", False) and "cholesky" in method and act_calib_debug_payload_lr:
        debug_dir = getattr(args, "act_calib_debug_dir", None) or getattr(args, "save_path", ".")
        os.makedirs(debug_dir, exist_ok=True)
        _sn = getattr(args, "save_name", "debug")
        dbg_path = os.path.join(debug_dir, f"act_calib_debug_calib_{_sn}_lowres.pt")
        torch.save(act_calib_debug_payload_lr, dbg_path)
        n_bytes = os.path.getsize(dbg_path)
        logging.info(
            "act_calib_debug (low-resource): saved calibration tensors for %d modules to %s (~%.2f MiB)",
            len(act_calib_debug_payload_lr),
            dbg_path,
            n_bytes / (1024 * 1024),
        )
    all_scaling_diag_matrix = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _is_lm_layer_module_name(name): # here only handle lm layers
            try:
                all_scaling_diag_matrix[name] = module.scaling_diag_matrixS
            except:
                all_scaling_diag_matrix[name] = None
            
    logging.info(f"Saving calibration cache to {cache_file}...")
    torch.save(all_scaling_diag_matrix, cache_file)
    logging.info("Calibration cache saved successfully!")
