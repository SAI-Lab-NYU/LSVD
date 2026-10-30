import datetime
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn

from utils import (
    utils,
    model_utils,
    data_utils,
    quant_utils,
    rotation_utils,
    gptq_utils,
    hadamard_utils,
    svd_utils,
    profile_utils,
    qwen_vlm_utils,
)
from utils.main_runtime_utils import get_rank_and_world_size, load_env, _run_lm_svd


def _move_qwen2_embed_tokens_to_dev(model, device):
    """
    Qwen2-VL nesting differs across transformers versions.
    Move whichever embed_tokens module exists onto the active device.
    """
    if hasattr(model, "language_model") and hasattr(model.language_model, "embed_tokens"):
        model.language_model.embed_tokens = model.language_model.embed_tokens.to(device)
        return
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "embed_tokens"):
        model.model.language_model.embed_tokens = model.model.language_model.embed_tokens.to(device)
        return
    raise AttributeError("Cannot locate embed_tokens for Qwen2-VL model.")


def _build_qwen2_inputs(message, image_processor, device):
    return qwen_vlm_utils.build_qwen2_eval_inputs(message, image_processor, device)


@torch.no_grad()
def _profile(dataloader, args, model, image_processor, tokenizer, out=False):
    inps = []
    outs = [] if out else None
    cache = {"i": 0, "attention_mask": [], "position_embeddings": []}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, *args, **kwargs):
            inps.append(inp)
            cache["i"] += 1
            raise ValueError

    class Catcherout(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, *args, **kwargs):
            if cache["i"] == len(inps):
                inps.append(inp)
            else:
                inps[cache["i"]] = inp
            inp = self.module(inp)
            outs.append(inp)
            cache["i"] += 1
            raise ValueError

    layers = model.model.visual.blocks
    model.model.visual = model.model.visual.to(utils.get_dev())
    layer = model.model.visual.merger.mlp
    layer = layer.to(utils.get_dev())

    if out:
        layer[0] = Catcherout(layer[0])
    else:
        layer[0] = Catcher(layer[0])

    _move_qwen2_embed_tokens_to_dev(model, utils.get_dev())

    for batch in dataloader:
        try:
            inputs = _build_qwen2_inputs(batch, image_processor, utils.get_dev())
            model.generate(
                **inputs,
                max_new_tokens=1,
                use_cache=True,
            )
        except ValueError:
            pass

    layer[0] = layer[0].module
    layer = layer.cpu()
    layers = layers.cpu()
    if out:
        return inps, outs
    return inps


def main(args):
    rank, world_size = get_rank_and_world_size()
    model, tokenizer, image_processor = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    modeldtype = model.dtype
    model.model.visual.to(modeldtype)
    args.pre_svd_rotate = False
    
    if args.profile_method:
        logging.warning(
            "Qwen2-VL profile_method is not implemented with Qwen2-specific hooks yet; disabling profile capture."
        )
        args.profile_method = False

    if args.svd_lm and not args.beta_then_svd:
        _run_lm_svd(model, args, tokenizer, image_processor)

    if args.vit_module:
        if not args.vit_online:
            trainloader = data_utils.get_loaders(
                args.cal_dataset,
                nsamples=args.vitnsamples,
                seed=args.seed,
                model=args.model,
                seqlen=model.seqlen,
                eval_mode=False,
            )
            dataloader, _ = trainloader
            utils.set_seed(args.seed)
            del trainloader
            inpsfp16 = _profile(dataloader, args, model, image_processor, tokenizer)
        config = model.model.config.vision_config

        if args.rotate:
            if args.vit_online:
                rotation_utils.wrap_layer_normvit(model)
                if args.mm_rh:
                    Q, Qmm = rotation_utils.rotate_modelvitmmonlineR_mmR(model, args)
                else:
                    Q = rotation_utils.rotate_modelvitmmonlineR(model, args)
                if args.vit_mmoff:
                    logging.info("mm proj rotation check again, skip them")
                utils.cleanup_memory(verbos=True)
            else:
                logging.warning("Qwen2-VL offline visual rotation is not validated; skipping visual rotation path.")
                Q = None
                Qmm = None
            quant_utils.add_actquant(model.model.visual)
            quant_utils.add_actquant(model.model.visual.merger.mlp)
            qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])

            for name in qlayers:
                if "fc2" in name and Q is not None:
                    qlayers[name].online_random_had = True
                    qlayers[name].had_K = Q
                    qlayers[name].K = 1
                    qlayers[name].fp32_had = args.fp32_had
                if "out_proj" in name:
                    had_K, K = hadamard_utils.get_hadK(config.num_attention_heads)
                    qlayers[name].online_partial_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].had_dim = config.hidden_size // config.num_attention_heads
                    qlayers[name].fp32_had = args.fp32_had
                if "visual.merger.mlp.0" in name and args.vit_online and not args.vit_mmoff:
                    logging.info("use mmproj input[0] online rotation")
                    had_K, K = hadamard_utils.get_hadK(model.config.vision_config.hidden_size)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                if "visual.merger.mlp.2" in name and not args.vit_mmoff:
                    if args.mm_rh and Qmm is not None:
                        logging.info("use mmproj input[2] online RH rotation")
                        qlayers[name].online_random_had = True
                        qlayers[name].had_K = Qmm
                        qlayers[name].K = 1
                        qlayers[name].fp32_had = args.fp32_had
                    else:
                        logging.info("use mmproj input[2] online Had rotation")
                        had_K, K = hadamard_utils.get_hadK(model.config.text_config.hidden_size)
                        qlayers[name].online_full_had = True
                        qlayers[name].had_K = had_K
                        qlayers[name].K = K
                        qlayers[name].fp32_had = args.fp32_had
        else:
            quant_utils.add_actquant(model.model.visual)
            quant_utils.add_actquant(model.model.visual.merger.mlp)

        if args.w_bits < 16:
            save_dict = {}
            if args.load_qmodel_path:
                assert args.rotate, "Model should be rotated to load a quantized model!"
                assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
                print("Load quantized model from ", args.load_qmodel_path)
                save_dict = torch.load(args.load_qmodel_path)
                model.load_state_dict(save_dict["model"])

            elif not args.w_rtn:
                logging.warning("Qwen2-VL visual GPTQ is not implemented; skipping visual weight quantization because w_rtn is disabled.")
            else:
                quantizers = gptq_utils.rtn_fwrdqwen2vit(model, utils.get_dev(), args)
                save_dict["w_vitquantizers"] = quantizers
                if not args.vit_mmoff:
                    quantizers = gptq_utils.rtn_fwrdqwen2mm(model, utils.get_dev(), args)
                    save_dict["w_mmquantizers"] = quantizers

            if args.save_qmodel_path:
                save_dict["model"] = model.state_dict()
                torch.save(save_dict, args.save_qmodel_path)

        if args.a_bits < 16 or args.v_bits < 16:
            logging.info(f"setting a clip ratio in vit {min(args.a_clip_ratio, args.vita_clip_ratio)}")
            qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
            down_proj_groupsize = -1
            if args.a_groupsize > 0 and ("llama" in args.model or "qwen" in args.model.lower()):
                down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
            for name in qlayers:
                layer_input_bits = args.a_bits
                layer_groupsize = args.a_groupsize
                layer_a_sym = not args.a_asym
                layer_a_clip = min(args.a_clip_ratio, args.vita_clip_ratio)
                if "lm_head" in name:
                    layer_input_bits = 16

                if "fc2" in name:
                    if args.int8_down_proj:
                        layer_input_bits = 8
                    layer_groupsize = down_proj_groupsize

                if "visual.merger.mlp" in name and args.vit_mmoff:
                    logging.info("mm projector act in 16 bit")
                    layer_input_bits = 16
                qlayers[name].quantizer.configure(
                    bits=layer_input_bits,
                    groupsize=layer_groupsize,
                    sym=layer_a_sym,
                    clip_ratio=layer_a_clip,
                )
        if not args.vit_online:
            inpsquant = _profile(dataloader, args, model, image_processor, tokenizer)
        if world_size > 1:
            dist.barrier()
        if not args.nobeta and not args.vit_online and args.rotate:
            logging.warning("Qwen2-VL beta fusion for offline visual rotation is not validated; skipping beta correction.")

    if args.svd_emb:
        svd_utils.svd_emb(model, args)

    if args.svd_lm and args.beta_then_svd:
        _run_lm_svd(model, args, tokenizer, image_processor)

    if not args.lm_off:
        if args.rotate:
            model = model.cpu()
            if args.svd_lm:
                rotation_utils.fuse_layer_norms_noebsvdqkv(model, args.svd_modules)
                rotation_utils.rotate_modelllavasvdqkv(model, args)
            else:
                rotation_utils.fuse_layer_norms_noeb(model)
                rotation_utils.rotate_modelllava(model, args)
            utils.cleanup_memory(verbos=True)
            quant_utils.add_actquant(model_utils.get_layers(model))
            qlayers = quant_utils.find_qlayers(model)
            for name in qlayers:
                if "down_proj" in name:
                    if args.svd_modules in ["all", "mlp"] and "ALinear" in name:
                        continue
                    had_K, K = hadamard_utils.get_hadK(model.config.text_config.intermediate_size)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                if "o_proj" in name:
                    had_K, K = hadamard_utils.get_hadK(model.config.text_config.num_attention_heads)
                    qlayers[name].online_partial_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].had_dim = (
                        model.config.text_config.hidden_size
                        // model.config.text_config.num_attention_heads
                    )
                    qlayers[name].fp32_had = args.fp32_had
        else:
            if args.w_bits >= 16 and args.a_bits >= 16 and args.v_bits >= 16 and args.k_bits >= 16:
                quant_utils.add_actquant(model_utils.get_layers(model))

        if args.profile_method:
            if args.vit_online:
                trainloader = data_utils.get_loaders(
                    args.cal_dataset,
                    nsamples=args.vitnsamples,
                    seed=args.seed,
                    model=args.model,
                    seqlen=model.seqlen,
                    eval_mode=False,
                )
                dataloader, _ = trainloader
                utils.set_seed(args.seed)
                del trainloader
            layer_ids = [0, 1, 2, 20, 21, 22, 37, 38, 39]
            profile_utils.save_layer(model, layer_ids, "before_weight_quant", args)
            activations = {}
            hooks = profile_utils.register_detailed_hooksv2svdqkvout(model, activations)
            profile_utils._profile(dataloader, args, model, image_processor, tokenizer)
            torch.save(activations, f"{args.save_path}/before_quant_act.pth")
            del activations
            profile_utils.remove_hooks(hooks)

        if args.w_bits < 16:
            save_dict = {}
            if args.load_qmodel_path:
                assert args.rotate, "Model should be rotated to load a quantized model!"
                assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
                print("Load quantized model from ", args.load_qmodel_path)
                save_dict = torch.load(args.load_qmodel_path)
                model.load_state_dict(save_dict["model"])

            elif not args.w_rtn:
                logging.warning("Qwen2-VL LM GPTQ is not implemented; skipping LM weight quantization because w_rtn is disabled.")
            else:
                quantizers = gptq_utils.rtn_fwrd(model, utils.get_dev(), args)
                save_dict["w_quantizers"] = quantizers

            if args.save_qmodel_path:
                save_dict["model"] = model.state_dict()
                torch.save(save_dict, args.save_qmodel_path)

        if args.a_bits < 16 or args.v_bits < 16:
            logging.info(f"setting a clip ratio in lm {min(args.a_clip_ratio, args.lma_clip_ratio)}")
            qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
            down_proj_groupsize = -1
            if args.a_groupsize > 0 and "qwen" in args.model.lower():
                down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
            for name in qlayers:
                layer_input_bits = args.a_bits
                layer_groupsize = args.a_groupsize
                layer_a_sym = not args.a_asym
                layer_a_clip = min(args.a_clip_ratio, args.lma_clip_ratio)
                if "v_proj" in name and args.v_bits < 16:
                    qlayers[name].out_quantizer.configure(
                        bits=args.v_bits,
                        groupsize=args.v_groupsize,
                        sym=not args.v_asym,
                        clip_ratio=args.v_clip_ratio,
                    )

                if "lm_head" in name:
                    layer_input_bits = 16

                if "down_proj" in name:
                    if args.int8_down_proj:
                        layer_input_bits = 8
                    layer_groupsize = down_proj_groupsize

                if "BLinear" in name and args.is_per_head_svd:
                    logging.info("Set up latent actquant using out_actquant")
                    qlayers[name].out_quantizer.configure(
                        bits=layer_input_bits,
                        groupsize=layer_groupsize,
                        sym=layer_a_sym,
                        clip_ratio=layer_a_clip,
                    )

                qlayers[name].quantizer.configure(
                    bits=layer_input_bits,
                    groupsize=layer_groupsize,
                    sym=layer_a_sym,
                    clip_ratio=layer_a_clip,
                )

        if args.k_bits < 16:
            if args.k_pre_rope:
                raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
            rope_function_name = model_utils.get_rope_function_name(model)
            layers = model_utils.get_layers(model)
            k_quant_config = {
                "k_bits": args.k_bits,
                "k_groupsize": args.k_groupsize,
                "k_sym": not args.k_asym,
                "k_clip_ratio": args.k_clip_ratio,
            }
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    **k_quant_config,
                )
            logging.info("finish qk rotation and k quant")

    if args.profile_method:
        activations = {}
        hooks = profile_utils.register_detailed_hooksv2svdqkvout(model, activations)
        profile_utils._profile(dataloader, args, model, image_processor, tokenizer)
        torch.save(activations, f"{args.save_path}/after_AWquant_act.pth")
        del activations
        profile_utils.remove_hooks(hooks)
        logging.info("Skip evaluation for profile calibrated results")
        return

    if getattr(args, "export_fused_qwen", False):
        # LSVD release: fused-checkpoint export (utils/export_utils.py) is not included.
        raise NotImplementedError("--export_fused_qwen is not included in the LSVD release.")

    testloader = data_utils.get_loaders(
        args.eval_dataset,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        hf_token=args.hf_token,
        eval_mode=True,
    )

    if "scienceqa" in args.eval_dataset.lower():
        from utils import eval_sqa_qwen2

        dataset_ppl = eval_sqa_qwen2.evaluator(
            model, testloader, utils.get_dev(), args, tokenizer, image_processor
        )
    elif "mathvista" in args.eval_dataset.lower():
        # LSVD release: MathVista evaluation (utils/eval_mathvista_qwen2.py) is not included.
        raise NotImplementedError("MathVista evaluation is not included in the LSVD release.")
    elif "vizwiz" in args.eval_dataset.lower():
        from utils import eval_utilsdistqwen2

        dataset_ppl = eval_utilsdistqwen2.evaluator(
            model, testloader, utils.get_dev(), args, tokenizer, image_processor
        )
    elif "seedbench" in args.eval_dataset.lower():
        from utils import eval_utilsdistqwen2

        dataset_ppl = eval_utilsdistqwen2.evaluator(
            model, testloader, utils.get_dev(), args, tokenizer, image_processor
        )
    elif "mmbench" in args.eval_dataset.lower():
        from utils import eval_utilsdistqwen2

        dataset_ppl = eval_utilsdistqwen2.evaluator(
            model, testloader, utils.get_dev(), args, tokenizer, image_processor
        )
    elif "ocrbench" in args.eval_dataset.lower():
        from utils import eval_utilsdistqwen2

        dataset_ppl = eval_utilsdistqwen2.evaluator(
            model, testloader, utils.get_dev(), args, tokenizer, image_processor
        )
    else:
        from utils import eval_utilsdistqwen2

        dataset_ppl = eval_utilsdistqwen2.evaluator(
            model, testloader, utils.get_dev(), args, tokenizer, image_processor
        )
    if args.wandb:
        import wandb

        wandb.log({"ppl/{}".format(args.eval_dataset.upper()): dataset_ppl})


if __name__ == "__main__":
    args = load_env()
    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        local_rank = os.environ.get("LOCAL_RANK", 0)
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=int(os.environ.get("DIST_TIMEOUT", 3600))),
        )
    main(args)
    if world_size > 1:
        dist.destroy_process_group()
