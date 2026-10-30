import logging
import os
import datetime

import torch
import torch.distributed as dist

from . import hadamard_utils, model_utils, quant_utils, svd_utils, utils


def get_rank_and_world_size():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


def initialize_distributed_for_evaluation():
    """Initialize an optional process group only after model reconstruction.

    The single-GPU capability runner launches several evaluator processes that
    share CUDA device 0.  Keeping their rank/world-size in deferred variables
    prevents PyTorch/Transformers from constructing DTensors while each worker
    rebuilds and strictly loads the quantized checkpoint.  The real distributed
    environment is promoted immediately before VLMEvalKit generation.
    """
    deferred_world_size = os.environ.get("DEFERRED_DIST_WORLD_SIZE")
    deferred_rank = os.environ.get("DEFERRED_DIST_RANK")
    if (deferred_world_size is None) != (deferred_rank is None):
        raise RuntimeError(
            "DEFERRED_DIST_WORLD_SIZE and DEFERRED_DIST_RANK must be set together"
        )

    if deferred_world_size is not None:
        if dist.is_initialized():
            raise RuntimeError(
                "Cannot promote a deferred distributed environment after a "
                "process group has already been initialized"
            )
        world_size = int(deferred_world_size)
        rank = int(deferred_rank)
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(
                f"Invalid deferred distributed rank {rank}/{world_size}"
            )
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
    else:
        rank, world_size = get_rank_and_world_size()

    if world_size > 1 and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        ready_file = os.environ.get("DEFERRED_DIST_READY_FILE")
        if ready_file:
            # The capability launcher uses this marker to reconstruct large
            # models one worker at a time, limiting host-RAM peaks.  Atomic
            # replacement ensures the launcher never observes a partial file.
            temporary_ready_file = f"{ready_file}.tmp.{os.getpid()}"
            with open(temporary_ready_file, "w", encoding="utf-8") as handle:
                handle.write(f"rank={rank} pid={os.getpid()}\n")
            os.replace(temporary_ready_file, ready_file)
        dist.init_process_group(
            backend=os.environ.get("DIST_BACKEND", "nccl"),
            timeout=datetime.timedelta(
                seconds=int(os.environ.get("DIST_TIMEOUT", 3600))
            ),
        )
    return rank, world_size


def load_env():
    args = utils.parser_gen()
    if args.wandb:
        import wandb

        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)
    utils.set_seed(args.seed)
    if args.vitnsamples == 0:
        args.vitnsamples = args.nsamples
        logging.info(f"set vitnsamples to nsamples{args.nsamples}")
    return args


def _lm_model_config(model):
    config = model.config
    return config.text_config if hasattr(config, "text_config") else config


def _lm_online_hadamard_for_svd_calibration(model, args):
    """Wrap LM o_proj/down_proj so pre-SVD calibration runs the matched rotated forward."""
    config = _lm_model_config(model)
    for layer in model_utils.get_layers(model):
        if isinstance(layer.mlp.down_proj, torch.nn.Linear):
            layer.mlp.down_proj = quant_utils.ActQuantWrapper(layer.mlp.down_proj)
        if isinstance(layer.self_attn.o_proj, torch.nn.Linear):
            layer.self_attn.o_proj = quant_utils.ActQuantWrapper(layer.self_attn.o_proj)

        if isinstance(layer.mlp.down_proj, quant_utils.ActQuantWrapper):
            had_K, K = hadamard_utils.get_hadK(config.intermediate_size)
            layer.mlp.down_proj.online_full_had = True
            layer.mlp.down_proj.had_K = had_K
            layer.mlp.down_proj.K = K
            layer.mlp.down_proj.fp32_had = args.fp32_had

        if isinstance(layer.self_attn.o_proj, quant_utils.ActQuantWrapper):
            had_K, K = hadamard_utils.get_hadK(config.num_attention_heads)
            layer.self_attn.o_proj.online_partial_had = True
            layer.self_attn.o_proj.had_K = had_K
            layer.self_attn.o_proj.K = K
            layer.self_attn.o_proj.had_dim = config.hidden_size // config.num_attention_heads
            layer.self_attn.o_proj.fp32_had = args.fp32_had


def _unwrap_pre_svd_lm_wrappers(model):
    for layer in model_utils.get_layers(model):
        if isinstance(layer.self_attn.o_proj, quant_utils.ActQuantWrapper):
            layer.self_attn.o_proj = layer.self_attn.o_proj.module
        if isinstance(layer.mlp.down_proj, quant_utils.ActQuantWrapper):
            layer.mlp.down_proj = layer.mlp.down_proj.module


def _run_lm_svd(model, args, tokenizer, image_processor):
    need_gptsvd_snap = bool(getattr(args, "gpt_svd_post_quant", False)) and int(
        getattr(args, "w_bits", 16)
    ) < 16
    if need_gptsvd_snap and not hasattr(model, "_gptsvd_original_weights"):
        # LSVD release: GPT-SVD post-quantization (utils/gpt_svd_utils.py) is not included.
        raise NotImplementedError("--gpt_svd_post_quant is not included in the LSVD release.")
    original_svd_modules = args.svd_modules
    if args.svd_modules == "all_sep":
        args.svd_modules = "attn"
        svd_utils.svd_lm_setup(model, args, tokenizer, image_processor)
        args.svd_modules = "mlp"
        svd_utils.svd_lm_setup(model, args, tokenizer, image_processor)
    elif args.svd_modules == "all_sep_mlpfine":
        args.svd_modules = "attn"
        svd_utils.svd_lm_setup(model, args, tokenizer, image_processor)
        args.svd_modules = "gaup"
        svd_utils.svd_lm_setup(model, args, tokenizer, image_processor)
        args.svd_modules = "down"
        svd_utils.svd_lm_setup(model, args, tokenizer, image_processor)
    elif args.svd_modules == "all_sep_fine":
        args.svd_modules = "qkv"
        svd_utils.svd_lm_setup(model, args, tokenizer, image_processor)
        args.svd_modules = "o"
        svd_utils.svd_lm_setup(model, args, tokenizer, image_processor)
        args.svd_modules = "gaup"
        svd_utils.svd_lm_setup(model, args, tokenizer, image_processor)
        args.svd_modules = "down"
        svd_utils.svd_lm_setup(model, args, tokenizer, image_processor)
    else:
        svd_utils.svd_lm_setup(model, args, tokenizer, image_processor)
    args.svd_modules = original_svd_modules
    _unwrap_pre_svd_lm_wrappers(model)
