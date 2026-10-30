import torch
import torch.nn as nn
import transformers
import logging
import os
import torch.distributed as dist
import datetime

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
)
from utils.main_runtime_utils import get_rank_and_world_size, load_env, _run_lm_svd

@torch.no_grad()
def _profile(dataloader, args, model, image_processor, tokenizer, out=False):
    dtype = next(iter(model.parameters())).dtype
    inps = []
    outs = torch.zeros(
        (args.vitnsamples, model.model.vision_model.num_patches, model.model.vision_model.config.hidden_size * 5),
        dtype=dtype,
        device=utils.get_dev(),
    ) if out else None
    cache = {'i': 0, 'attention_mask': [], 'position_embeddings':[]}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inps.append(inp)
            cache['i'] += 1
            raise ValueError
    class Catcherout(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inps[cache['i']] = inp
            inp = self.module(inp)
            outs[cache['i']] = inp
            cache['i'] += 1
            raise ValueError
    
    layers = model.model.vision_model.encoder.layers
    model.model.vision_model = model.model.vision_model.to(utils.get_dev())
    projector = model.model.connector.modality_projection.proj
    projector = projector.to(utils.get_dev())

    
    if out:
        model.model.connector.modality_projection.proj = Catcherout(projector)
    else:
        model.model.connector.modality_projection.proj = Catcher(projector)

    model.model.text_model.embed_tokens = model.model.text_model.embed_tokens.to(utils.get_dev())

    for batch in dataloader:
        try:
            inputs = gptq_utils.message_to_promptsmolvlm(batch, image_processor, model, tokenizer)
            model.generate(
                **inputs,
                max_new_tokens=1,
                use_cache=True,
            )
        except ValueError:
            pass

    model.model.connector.modality_projection.proj = model.model.connector.modality_projection.proj.module
    projector = projector.cpu()
    layers = layers.cpu()
    if out:
        return inps, outs
    return inps


def main(args):
    rank, world_size = get_rank_and_world_size()
    model, image_processor = model_utils.get_model(args.model, args.hf_token)
    tokenizer = None
    model.eval()
    modeldtype = model.dtype
    model.model.vision_model.to(modeldtype)
    print(model.model.vision_model.dtype)
    args.pre_svd_rotate = False

    hybrid_ffn_pending = bool(args.svd_lm and args.use_hybrid_ffn)

    if args.svd_lm and not args.beta_then_svd:
        if args.svd_modules is not None:
            _run_lm_svd(model, args, tokenizer, image_processor)

    if args.vit_module:
        ### todo add vit learning whole process here
        if not args.vit_online:
            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.vitnsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False
            )
            dataloader, _ = trainloader
            utils.set_seed(args.seed) # do we need this?
            del trainloader
            inpsfp16= _profile(dataloader, args, model, image_processor, tokenizer)
        config = model.model.vision_model.config
        
        if args.rotate:
            if args.vit_online:
                rotation_utils.wrap_layer_normvit(model) # just ignore for now
                if args.mm_rh:
                    Q, Qmm = rotation_utils.rotate_modelvitmmonlineR_mmR(model, args)
                    logging.info('mmrh not supported yet')
                else:
                    Q, Qmm = rotation_utils.rotate_smovlmvitmmonlineR(model, args)
                if args.vit_mmoff:
                    logging.info('vit mm off not supported yet')
                utils.cleanup_memory(verbos=True)
            else:
                logging.info('not supported yet')
                rotation_utils.fuse_layer_normsvit_returnskip(model)
                qlayers = model.model.vision_model # here add online hadamard for pre-ln
                qlayers.pre_layrnorm = model_utils.LNRotWrapper(qlayers.pre_layrnorm) # centering for vit
                if args.mm_rh:
                    Q, Qmm = rotation_utils.rotate_modelvitmmv2(model, args)
                else:
                    Q = rotation_utils.rotate_modelvitmmv2(model, args)
                utils.cleanup_memory(verbos=True)
            quant_utils.add_actquant(model.model.vision_model.encoder.layers) #Add Activation Wrapper to the model
            if not args.vit_mmoff:
                quant_utils.add_actquant(model.model.connector)
            qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
            
            for name in qlayers:# for clip, todo here
                if 'fc2' in name:
                    qlayers[name].online_random_had = True
                    qlayers[name].had_K = Q
                    qlayers[name].K = 1
                    qlayers[name].fp32_had = args.fp32_had
                if 'out_proj' in name: # why need this if already done ov_rotate in rotate_model
                    had_K, K = hadamard_utils.get_hadK(config.num_attention_heads)
                    qlayers[name].online_partial_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].had_dim = config.hidden_size//config.num_attention_heads
                    qlayers[name].fp32_had = args.fp32_had
                if 'connector' in name and args.vit_online and not args.vit_mmoff:
                    logging.info('use mmproj input RH rotation')
                    qlayers[name].online_random_had = True 
                    qlayers[name].had_K = Qmm
                    qlayers[name].K = 1
                    qlayers[name].fp32_had = args.fp32_had
        else:
            quant_utils.add_actquant(model.model.vision_model.encoder.layers) #Add Activation Wrapper to the model
            quant_utils.add_actquant(model.model.connector)
        
        #### add vit weight quantization
        if args.w_bits < 16:
            save_dict = {}
            if args.load_qmodel_path: # Load Quantized Rotated Model
                assert args.rotate, "Model should be rotated to load a quantized model!"
                assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
                print("Load quantized model from ", args.load_qmodel_path)
                save_dict = torch.load(args.load_qmodel_path)
                model.load_state_dict(save_dict["model"])
                
            elif not args.w_rtn: # GPTQ Weight Quantization
                raise NotImplementedError("SmolVLM GPTQ not supported")
            else: # RTN Weight Quantization
                quantizers = gptq_utils.rtn_fwrdsmovlmvit(model, utils.get_dev(), args)
                save_dict["w_vitquantizers"] = quantizers
                if not args.vit_mmoff:
                    quantizers = gptq_utils.rtn_fwrdsmovlmmm(model, utils.get_dev(), args)
                    save_dict["w_mmquantizers"] = quantizers
            if args.save_qmodel_path:
                save_dict["model"] = model.state_dict()
                torch.save(save_dict, args.save_qmodel_path)

        # Add vit Input Quantization
        if args.a_bits < 16 or args.v_bits < 16:
            logging.info(f'setting a clip ratio in vit {min(args.a_clip_ratio, args.vita_clip_ratio)}')
            qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
            down_proj_groupsize = -1
            if args.a_groupsize > 0 and ("llama" in args.model or 'llava' in args.model): # here change as well  for all
                down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
            for name in qlayers:
                layer_input_bits = args.a_bits
                layer_groupsize = args.a_groupsize
                layer_a_sym = not(args.a_asym)
                layer_a_clip = min(args.a_clip_ratio, args.vita_clip_ratio)
                if 'lm_head' in name: #Skip lm_head quantization   
                    layer_input_bits = 16
                
                if 'fc2' in name: #Set the down_proj precision
                    if args.int8_down_proj:
                        layer_input_bits = 8
                    layer_groupsize = down_proj_groupsize

                if 'connector' in name and args.vit_mmoff:
                    logging.info(f"connector act in {16} bit")
                    layer_input_bits = 16 # skip vit mm act quant
                qlayers[name].quantizer.configure(bits=layer_input_bits,
                                                groupsize=layer_groupsize,
                                                sym=layer_a_sym,
                                                clip_ratio=layer_a_clip)
        if not args.vit_online:
            inpsquant = _profile(dataloader, args, model, image_processor, tokenizer) 
        if world_size > 1:
            dist.barrier()
        if not args.nobeta and not args.vit_online and args.rotate:
            from utils import beta_utils
            try:
                beta_path=None
                linearbias = torch.load(beta_path)
                logging.info(f"use beta from {beta_path}")
            except:
                weight = None
                bias_ = None
                Q = model.model.vision_model.pre_layrnorm.had_K
                linearbias = beta_utils.train_bias_linear(inpsfp16, inpsquant, args.beta_epochs, args.beta_lr, Q, args=args, weight=weight, bias=bias_)
            beta_utils.fuse_linearbias(linearbias, model, is_mm=True)
            del linearbias
            del inpsfp16
            del inpsquant

    if args.svd_emb:
        svd_utils.svd_emb(model, args)

    hybrid_ffn_pending = bool(args.svd_lm and args.use_hybrid_ffn)

    if args.svd_lm and args.beta_then_svd:
        if args.svd_modules is not None:
            _run_lm_svd(model, args, tokenizer, image_processor)

    if not args.lm_off:
        # Rotate the weights
        if args.rotate:
            model = model.cpu()
            if args.svd_lm:
                rotation_utils.fuse_layer_norms_noebsvdqkv(model, args.svd_modules)
                rotation_utils.rotate_modelsmolvlmsvdqkv(model, args)
            else:
                rotation_utils.fuse_layer_norms_noeb(model)
                rotation_utils.rotate_modelsmolvlm(model, args)
            utils.cleanup_memory(verbos=True)
            if args.use_hybrid_ffn:
                for layer in model_utils.get_layers(model):
                    quant_utils.add_actquant(layer.self_attn)
            else:
                quant_utils.add_actquant(model.model.text_model.layers) #Add Activation Wrapper to only the LM model
            qlayers = quant_utils.find_qlayers(model)
            for name in qlayers:# for clip, todo here
                if 'down_proj' in name:
                    had_K, K = hadamard_utils.get_hadK(model.model.text_model.config.intermediate_size)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                if 'o_proj' in name:
                    had_K, K = hadamard_utils.get_hadK(model.model.text_model.config.num_attention_heads)
                    qlayers[name].online_partial_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].had_dim = model.model.text_model.config.hidden_size//model.model.text_model.config.num_attention_heads
                    qlayers[name].fp32_had = args.fp32_had
        else:
            if args.w_bits >= 16 and args.a_bits >= 16 and args.v_bits >= 16 and args.k_bits >= 16:
                if args.use_hybrid_ffn:
                    for layer in model_utils.get_layers(model):
                        quant_utils.add_actquant(layer.self_attn)
                else:
                    quant_utils.add_actquant(model.model.text_model.layers) #Add Activation Wrapper to the model as the rest of the code assumes it is present

        if hybrid_ffn_pending:
            from utils.hybrid_ffn_utils import hybrid_ffn_setup
            for layer in model_utils.get_layers(model):
                quant_utils.add_actquant(layer.mlp)
            hybrid_ffn_setup(model, args, tokenizer=tokenizer, image_processor=image_processor)
        
        if args.profile_method:
            if args.vit_online:
                trainloader = data_utils.get_loaders(
                    args.cal_dataset, nsamples=args.vitnsamples,
                    seed=args.seed, model=args.model,
                    seqlen=model.seqlen, eval_mode=False
                )
                dataloader, _ = trainloader
                utils.set_seed(args.seed)
                del trainloader
            layer_ids = [_ for _ in range(len(model_utils.get_layers(model)))]
            profile_utils.save_layer(model, layer_ids, 'before_weight_quant', args)
            activations = {}
            hooks = profile_utils.register_detailed_hooksv2(model, activations,) if not args.svd_lm else profile_utils.register_detailed_hooksv2svd(model, activations,)
            torch.save(profile_utils._profile(dataloader, args, model, image_processor, tokenizer), f"{args.save_path}/before_quant_output.pth")
            torch.save(activations, f"{args.save_path}/before_quant_act.pth")
            del activations
            profile_utils.remove_hooks(hooks)
        
        if args.w_bits < 16:
            save_dict = {}
            if args.load_qmodel_path: # Load Quantized Rotated Model
                assert args.rotate, "Model should be rotated to load a quantized model!"
                assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
                print("Load quantized model from ", args.load_qmodel_path)
                save_dict = torch.load(args.load_qmodel_path)
                model.load_state_dict(save_dict["model"])
                
            elif not args.w_rtn: # GPTQ Weight Quantization
                if args.use_hybrid_ffn:
                    raise NotImplementedError(
                        "use_hybrid_ffn currently supports LM weight quantization only with --w_rtn "
                        "in main_smolvlm_hybrid_ffn.py; the GPTQ path is not wired for HybridSVDQuantFFN yet."
                    )
                assert ("llama" in args.model or 'llava' in args.model), "Only llama/llava is supported for GPTQ!"
                
                trainloader = data_utils.get_loaders(
                    args.cal_dataset, nsamples=args.nsamples,
                    seed=args.seed, model=args.model,
                    seqlen=model.seqlen, eval_mode=False
                )
                quantizers = gptq_utils.gptq_fwrdllava(model, trainloader, utils.get_dev(), args, tokenizer, image_processor)
                save_dict["w_quantizers"] = quantizers
            else: # RTN Weight Quantization
                quantizers = gptq_utils.rtn_fwrd(model, utils.get_dev(), args) #, start_id=start_id
                save_dict["w_quantizers"] = quantizers
                
            if args.save_qmodel_path:
                save_dict["model"] = model.state_dict()
                torch.save(save_dict, args.save_qmodel_path)

        if args.profile_method:
            layer_ids = [_ for _ in range(len(model_utils.get_layers(model)))]
            profile_utils.save_layer(model, layer_ids, 'after_weight_quant', args)
            torch.save(profile_utils._profile(dataloader, args, model, image_processor, tokenizer), f"{args.save_path}/before_Aquant_output.pth")


        # Add Input Quantization
        if args.a_bits < 16 or args.v_bits < 16:
            logging.info(f'setting a clip ratio in lm {min(args.a_clip_ratio, args.lma_clip_ratio)}')
            qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
            down_proj_groupsize = -1
            if args.a_groupsize > 0 and "llama" in args.model:
                down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
            for name in qlayers:
                layer_input_bits = args.a_bits
                layer_groupsize = args.a_groupsize
                layer_a_sym = not(args.a_asym)
                layer_a_clip = min(args.a_clip_ratio, args.lma_clip_ratio)
                
                if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
                    qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                                groupsize=args.v_groupsize,
                                                sym=not(args.v_asym),
                                                clip_ratio=args.v_clip_ratio)
                
                if 'lm_head' in name: #Skip lm_head quantization   
                    layer_input_bits = 16
                
                if 'down_proj' in name: #Set the down_proj precision
                    if args.int8_down_proj:
                        layer_input_bits = 8
                    layer_groupsize = down_proj_groupsize

                is_hybrid_svd_blinear = (
                    'mlp.gate_svd.BLinear' in name or
                    'mlp.up_svd.BLinear' in name
                )
                if is_hybrid_svd_blinear or ('BLinear' in name and args.is_per_head_svd):
                    logging.info(f"Set up latent actquant using out_actquant for {name}")
                    qlayers[name].out_quantizer.configure(bits=layer_input_bits,
                                                groupsize=layer_groupsize,
                                                sym=layer_a_sym,
                                                clip_ratio=layer_a_clip)
                    
                qlayers[name].quantizer.configure(bits=layer_input_bits,
                                                groupsize=layer_groupsize,
                                                sym=layer_a_sym,
                                                clip_ratio=layer_a_clip)

        if args.k_bits < 16:
            if args.k_pre_rope:
                raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
            else:
                rope_function_name = model_utils.get_rope_function_name(model)
                layers = model_utils.get_layers(model)
                k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                            "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
                for layer in layers:
                    rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                                layer.self_attn, 
                                rope_function_name, 
                                config=model.model.text_model.config,
                                **k_quant_config)
                logging.info('finish qk rotation and k quant')
    
    if args.profile_method:
        activations = {}
        hooks = profile_utils.register_detailed_hooksv2(model, activations,) if not args.svd_lm else profile_utils.register_detailed_hooksv2svd(model, activations,)
        torch.save(profile_utils._profile(dataloader, args, model, image_processor, tokenizer), f"{args.save_path}/after_AWquant_output.pth")
        torch.save(activations, f"{args.save_path}/after_AWquant_act.pth")
        del activations
        profile_utils.remove_hooks(hooks)
        
        logging.info("Skip evaluation for profile calibrated results")
        return
    # Evaluating on dataset
    testloader = data_utils.get_loaders(
            args.eval_dataset,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
            hf_token=args.hf_token,
            eval_mode=True
        )

    if 'scienceqa' in args.eval_dataset.lower():
        from utils import eval_utilsdistsmovlm
        dataset_ppl = eval_utilsdistsmovlm.evaluator(model, testloader, utils.get_dev(), args, image_processor)
    elif 'vizwiz' in args.eval_dataset.lower():
        # LSVD release: VizWiz evaluation (utils/eval_smolvlm_vizwiz.py) is not included.
        raise NotImplementedError("VizWiz evaluation is not included in the LSVD release.")
    elif 'seedbench' in args.eval_dataset.lower():
        from utils import eval_utilsdistsmovlm
        dataset_ppl = eval_utilsdistsmovlm.evaluator(model, testloader, utils.get_dev(), args, image_processor)
    elif 'mmbench' in args.eval_dataset.lower():
        from utils import eval_utilsdistsmovlm
        dataset_ppl = eval_utilsdistsmovlm.evaluator(model, testloader, utils.get_dev(), args, image_processor)
    else:
        from utils import eval_utilsdistsmovlm
        dataset_ppl = eval_utilsdistsmovlm.evaluator(model, testloader, utils.get_dev(), args, image_processor)
    if args.wandb:
        import wandb

        wandb.log({'ppl/{}'.format(args.eval_dataset.upper()): dataset_ppl})


if __name__ == '__main__':
    args = load_env() # args/seeds setup before distributed setup?
    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        local_rank = os.environ.get('LOCAL_RANK', 0)
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
        )
    # rank = dist.get_rank()
    # if dist.is_initialized():
    #     rank = dist.get_rank()
    #     DEV = torch.device(f'cuda:{rank}')
    # else:
    #     DEV = torch.device('cuda:0')
    # use utils.get_dev() instead
    main(args)
    if world_size > 1:
        dist.destroy_process_group()
