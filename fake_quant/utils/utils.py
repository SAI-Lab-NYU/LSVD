import argparse
import pprint
import torch
import random
import numpy as np
import os
from datetime import datetime
import logging


from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

supported_models = [
            'meta-llama/Llama-2-7b-hf',
            'meta-llama/Llama-2-13b-hf',
            'meta-llama/Llama-2-70b-hf',
            'meta-llama/Meta-Llama-3-8B',
            'meta-llama/Meta-Llama-3-70B',
            'facebook/opt-125m',
            'liuhaotian/llava-v1.5-7b',
            'liuhaotian/llava-v1.5-13b',
            'liuhaotian/llava-v1.6-vicuna-7b',
            'liuhaotian/llava-v1.6-vicuna-13b',
            'llava-hf/llava-v1.6-vicuna-7b-hf',
            'llava-hf/llava-v1.6-vicuna-13b-hf',
            "HuggingFaceTB/SmolVLM-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "OpenGVLab/InternVL3-8B",
            "OpenGVLab/InternVL3-14B",
            "OpenGVLab/InternVL3_5-8B-HF",
            "OpenGVLab/InternVL3_5-14B-HF",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "OpenGVLab/InternVL2-8B",
            "OpenGVLab/InternVL2_5-8B",
            "OpenGVLab/InternVL3-8B",
            "Qwen/Qwen3-VL-4B-Thinking",
            "Qwen/Qwen3-VL-8B-Thinking",
            "Qwen/Qwen3-VL-8B-Instruct",
            "allenai/Molmo-7B-O-0924",
            ]
supported_datasets = [
    'wikitext2', 'ptb', 'c4',
    'ScienceQA_Train','ScienceQA_TEST', 'VizWiz', 
    'SEEDBench', 'SEEDBench_IMG', 'SEEDBench2', 
    'OCRBench', 'MMBench_TEST_EN_V11', 
    'MathVista_MINI', 'HallusionBench', 'POPE', 'MMVet'
]

# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import torch.distributed as dist
def get_dev():
    if not torch.cuda.is_available():
        return torch.device('cpu')

    # Global distributed rank is not necessarily a CUDA device index: several
    # Gloo ranks may share one GPU and all receive LOCAL_RANK=0.
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        if local_rank < 0:
            raise ValueError(f'LOCAL_RANK must be non-negative; got {local_rank}')
        return torch.device(f'cuda:{local_rank}')
    if dist.is_initialized():
        return torch.device(f'cuda:{dist.get_rank()}')
    return torch.device('cuda')
DEV = get_dev()


def llama_down_proj_groupsize(model, groupsize):
    
    assert groupsize > 1, 'groupsize should be greater than 1!'
    
    if model.config.intermediate_size % groupsize == 0:
        logging.info(f'(Act.) Groupsiz = Down_proj Groupsize: {groupsize}')
        return groupsize

    group_num = int(model.config.hidden_size/groupsize)
    assert groupsize*group_num == model.config.hidden_size, 'Invalid groupsize for llama!'

    down_proj_groupsize = model.config.intermediate_size//group_num
    assert down_proj_groupsize*group_num == model.config.intermediate_size, 'Invalid groupsize for down_proj!'
    logging.info(f'(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}')
    return down_proj_groupsize


def set_seed(seed):
    np.random.seed(seed)  # Set NumPy seed
    random.seed(seed)  # Set Python's built-in random seed
    torch.manual_seed(seed)  # Set PyTorch seed for CPU & CUDA
    torch.cuda.manual_seed(seed)  # Set seed for CUDA (if using GPU)
    torch.cuda.manual_seed_all(seed)  # If multi-GPU, set for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-optimizations for determinism


# Dump the log both to console and a log file.
def config_logging(log_file, level=logging.INFO):
    class LogFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                self._style._fmt = "%(asctime)s - %(message)s"
            else:
                self._style._fmt = "%(asctime)s - %(levelname)s - %(message)s"
            return super().format(record)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LogFormatter())

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(LogFormatter())

    logging.basicConfig(level=level, handlers=[console_handler, file_handler])


def parser_gen():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='Model to load;', choices=supported_models)
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--eval_dataset', type=str, default='wikitext2',
                        help='Dataset for Evaluation (default: wikitext2)', choices=supported_datasets,)
    parser.add_argument(
        '--eval_datasets',
        nargs='+',
        default=None,
        choices=supported_datasets,
        help=(
            'Optional ordered list of datasets to evaluate sequentially after one '
            'model preparation/tuning pass. --eval_dataset remains the primary '
            'dataset used to derive the base output path.'
        ),
    )
    parser.add_argument(
        '--defer_vlm_scoring',
        action='store_true',
        help=(
            'Write merged VLM predictions/workbooks but defer dataset.evaluate() '
            'to a CPU-only scoring job.'
        ),
    )
    parser.add_argument(
        '--eval_max_new_tokens',
        type=int,
        default=None,
        help=(
            'Override the dataset-aware VLM generation limit. Defaults to 512 '
            'for MMVet and 16 for HallusionBench/POPE and existing MCQ tasks.'
        ),
    )
    parser.add_argument(
        '--vlm_judge_model',
        type=str,
        default=None,
        help=(
            'VLMEvalKit judge model used during scoring. Required to score '
            'MMVet unless --defer_vlm_scoring is set; HallusionBench and POPE '
            'default to deterministic exact yes/no extraction.'
        ),
    )
    parser.add_argument(
        '--vlm_judge_nproc',
        type=int,
        default=4,
        help='Number of CPU/API judge workers used by VLMEvalKit.',
    )
    parser.add_argument(
        '--skip_evaluation',
        action='store_true',
        help='Prepare/tune and optionally save the model, then exit before evaluation.',
    )
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--bsz', type=int, default=32,
                        help='Batch-size for PPL evaluation (default:32)')


    # Rotation Arguments
    parser.add_argument('--rotate', action=argparse.BooleanOptionalAction, default=False, 
                        help='''Rotate the moodel. This will include online rotation for down-projection and
                        out-projection. Note that this does not apply rotation to the K/Q and they will be rotated
                        if we want to quantize the Keys''')
    parser.add_argument('--rotate_mode', type=str, default='hadamard', choices=['hadamard', 'random'])
    parser.add_argument('--rotation_seed', type=int, default=-1,
                        help='Random Seed for generating random matrix!!')
    parser.add_argument('--fp32_had', action=argparse.BooleanOptionalAction, default=False,
                        help='Apply Hadamard rotation in FP32 (default: False)')
    parser.add_argument('--rot_lr', type=float, default=1,
                        help='learning rate for rotation!!')
    parser.add_argument('--sq_lr', type=float, default=1,
                        help='learning rate for scaling factor!!')
    parser.add_argument('--rot_epochs', type=int, default=100,
                        help='learning rate for rotation!!')
    parser.add_argument('--rot_dim', type=int, default=0,
                        help='dimension for rotation matrix!!')
    parser.add_argument('--sq', action=argparse.BooleanOptionalAction, default=False, 
                        help='weather use learnt smooothquant for learning rotation matrix!!')
    parser.add_argument('--beta_lr', type=float, default=1,
                        help='learning rate for beta!!')
    parser.add_argument('--beta_epochs', type=int, default=100,
                        help='learning rate for beta!!')
    parser.add_argument('--bs', type=int, default=512,
                        help='mini batchsize for beta!!')
    parser.add_argument('--nobeta',action=argparse.BooleanOptionalAction, default=False,  
        help='whether use beta learning/bias')
    # Activation Quantization Arguments
    parser.add_argument('--a_bits', type=int, default=16,
                        help='''Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)''')
    parser.add_argument('--a_bits_calib', type=int, default=None,
                        help='''Number of bits for inputs of the Linear layers in calibration for SVD-LLM style.''')
    parser.add_argument('--a_groupsize', type=int, default=-1, 
                        help='Groupsize for activation quantization. Note that this should be the same as w_groupsize')
    parser.add_argument('--a_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric Activation quantization (default: False)')
    parser.add_argument('--a_clip_ratio', type=float, default=1.0,
        help='Clip ratio for activation quantization. new_max = max * clip_ratio')
    parser.add_argument('--lma_clip_ratio', type=float, default=1.0,
        help='Clip ratio for lm activation quantization. new_max = max * clip_ratio')
    parser.add_argument('--vita_clip_ratio', type=float, default=1.0,
        help='Clip ratio for vit activation quantization. new_max = max * clip_ratio')
    # Weight Quantization Arguments
    parser.add_argument('--w_bits', type=int, default=16, 
                        help='Number of bits for weights of the Linear layers')
    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--w_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric weight quantization (default: False)')
    parser.add_argument('--w_rtn', action=argparse.BooleanOptionalAction, default=False,
                        help='Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ')
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=False,
                        help='''Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization''')
    parser.add_argument('--gpt_svd_post_quant', action=argparse.BooleanOptionalAction, default=False,
                        help='Use loss-aware SVD + residual-aware GPT-style post-SVD quantization for LM SVD modules.')
    parser.add_argument('--postsvd_bits_blinear', '--postsvd_bits_b', dest='postsvd_bits_blinear', type=int, default=None,
                        help='Bit-width for deployed BLinear.weight GPT-SVD post quantization.')
    parser.add_argument('--postsvd_bits_alinear', '--postsvd_bits_a', dest='postsvd_bits_alinear', type=int, default=None,
                        help='Bit-width for deployed ALinear.weight GPT-SVD post quantization.')
    parser.add_argument('--postsvd_group_blinear', '--postsvd_group_b', dest='postsvd_group_blinear', type=int, default=None,
                        help='Group size for deployed BLinear.weight GPT-SVD post quantization.')
    parser.add_argument('--postsvd_group_alinear', '--postsvd_group_a', dest='postsvd_group_alinear', type=int, default=None,
                        help='Group size for deployed ALinear.weight GPT-SVD post quantization.')
    parser.add_argument('--postsvd_damping_x', type=float, default=1e-6,
                        help='Input covariance damping for GPT-SVD post quantization.')
    parser.add_argument('--postsvd_damping_g', type=float, default=1e-6,
                        help='Output-gradient covariance damping for GPT-SVD post quantization.')
    parser.add_argument('--postsvd_damping_sz', type=float, default=1e-6,
                        help='Latent covariance damping for GPT-SVD post quantization.')
    parser.add_argument('--postsvd_damping_ma', type=float, default=1e-6,
                        help='A-step curvature damping for GPT-SVD post quantization.')
    parser.add_argument('--postsvd_clip_comp', type=float, default=None,
                        help='Optional clipping limit for mismatch compensation in GPT-SVD post quantization.')
    parser.add_argument('--postsvd_target_weight', type=str, default='original_dense',
                        choices=['original_dense', 'low_rank'],
                        help='Target matrix for GPT-SVD post quantization: pre-SVD original dense weight or current low-rank reconstruction.')
    parser.add_argument('--postsvd_compute_dtype', type=str, default='float32',
                        choices=['float32', 'float64'],
                        help='Compute dtype for GPT-SVD post quantization.')
    parser.add_argument('--postsvd_blocksize_blinear', '--postsvd_blocksize_b', dest='postsvd_blocksize_blinear', type=int, default=128,
                        help='Block size for GPT-SVD BLinear-step processing.')
    parser.add_argument('--postsvd_blocksize_alinear', '--postsvd_blocksize_a', dest='postsvd_blocksize_alinear', type=int, default=128,
                        help='Block size for GPT-SVD ALinear-step processing.')
    parser.add_argument('--postsvd_progress_interval', type=int, default=256,
                        help='Progress logging interval for GPT-SVD row processing.')
    parser.add_argument('--postsvd_log_rows', '--postsvd_log_columns', dest='postsvd_log_rows',
                        action=argparse.BooleanOptionalAction, default=False,
                        help='Record per-row GPT-SVD diagnostics.')
    parser.add_argument('--postsvd_profile_timing', action=argparse.BooleanOptionalAction, default=True,
                        help='Record GPT-SVD timing diagnostics.')
    parser.add_argument('--postsvd_disable_alinear_gpt_comp', action=argparse.BooleanOptionalAction, default=False,
                        help='Disable GPT compensation for ALinear and use row-wise naive quantization for that stage.')
    parser.add_argument('--postsvd_disable_blinear_gpt_comp', action=argparse.BooleanOptionalAction, default=False,
                        help='Disable GPT compensation for BLinear and use row-wise naive quantization for that stage.')
    parser.add_argument('--postsvd_use_naive_rowwise_only', action=argparse.BooleanOptionalAction, default=False,
                        help='Disable GPT compensation for both ALinear and BLinear and use naive row-wise quantization only.')
    parser.add_argument('--postsvd_objective_mode', type=str, default='sx_sg',
                        choices=['sx_sg', 'sx_only'],
                        help='Objective weighting for GPT-SVD post quantization: full Sx+Sg or Sx-only layer-output objective.')
    parser.add_argument('--postsvd_stage_plan', '--postsvd_stage_order', dest='postsvd_stage_plan', type=str, default='a_then_b',
                        choices=['a_then_b', 'b_then_a', 'b_then_a_then_b', 'a_then_b_then_a'],
                        help='Execution plan for GPT-SVD post quantization stages.')
    parser.add_argument('--postsvd_alinear_correction_scale', type=float, default=1.0,
                        help='Scale multiplier applied to ALinear GPT correction before quantization.')
    parser.add_argument('--postsvd_blinear_correction_scale', type=float, default=1.0,
                        help='Scale multiplier applied to BLinear GPT correction before quantization.')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--vitnsamples', type=int, default=0,
                        help='Number of calibration data samples for vit learn beta.')
    parser.add_argument('--cal_dataset', type=str, default='wikitext2',
                        help='calibration data samples for GPTQ.', choices=supported_datasets)
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action=argparse.BooleanOptionalAction, default=False,
                        help='act-order in GPTQ')
    parser.add_argument('--dosample', action=argparse.BooleanOptionalAction, default=False\
        , help='in llava, whether use do_sample')

    # General Quantization Arguments
    parser.add_argument('--int8_down_proj', action=argparse.BooleanOptionalAction, default=False,
                        help='Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8')

    # KV-Cache Quantization Arguments
    parser.add_argument('--v_bits', type=int, default=16,
                        help='''Number of bits for V-cache quantization. 
                        Note that quantizing the V-cache does not need any other rotation''')
    parser.add_argument('--v_groupsize', type=int, default=-1)
    parser.add_argument('--v_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric V-cache quantization')
    parser.add_argument('--v_clip_ratio', type=float, default=1.0,
        help='Clip ratio for v-cache quantization. new_max = max * clip_ratio')
    
    parser.add_argument('--k_bits', type=int, default=16,
                        help='''Number of bits for K-cache quantization. 
                        Note that quantizing the K-cache needs another rotation for the keys/queries''')
    parser.add_argument('--k_groupsize', type=int, default=-1)
    parser.add_argument('--k_asym', action=argparse.BooleanOptionalAction, default=False, 
                        help='ASymmetric K-cache quantization')
    parser.add_argument('--k_pre_rope', action=argparse.BooleanOptionalAction, default=False, 
                        help='Pre-RoPE quantization for K-cache (not Supported yet!)')
    parser.add_argument('--k_clip_ratio', type=float, default=1.0,
        help='Clip ratio for k-cache quantization. new_max = max * clip_ratio')

    # low rank parameters
    ### low rank experiments
    parser.add_argument('--is_ma_hack', action=argparse.BooleanOptionalAction, default=False,
        help='whether use MA hack for local ft')
    parser.add_argument('--topk', type=int, default=1,
        help='topk for super weight analysis')
    parser.add_argument('--smooth_grad', action=argparse.BooleanOptionalAction, default=False,
        help='whether smooth grad info')
    parser.add_argument('--smooth_method', type=str, default='mean', choices=['mean', 'zero'],
        help='smooth method for grad info')
    parser.add_argument('--smooth_percentile', type=float, default=-1,
        help='percentile for grad info')
    parser.add_argument('--smooth_outlier_std', type=float, default=-1,
        help='outlier std for grad info')
    parser.add_argument('--smooth_power_threshold', type=float, default=-1,
        help='power threshold for grad info')
    parser.add_argument('--smooth_ma_only', action=argparse.BooleanOptionalAction, default=False,
        help='whether use ma layer only smoothfor grad info')
    parser.add_argument('--smooth_final', action=argparse.BooleanOptionalAction, default=False,
        help='whether smooth grad info at final step or per get_layer_importance step')
    parser.add_argument('--fix_loss', action=argparse.BooleanOptionalAction, default=False,
        help='whether fix zigzag loss for local ft')
    parser.add_argument('--sw_analyze', action=argparse.BooleanOptionalAction, default=False,
        help='whether analyze super weight')
    parser.add_argument('--scheduler_type', type=str, default=None, choices=[None, 'cosine','step'],
        help='scheduler type for local ft')
    parser.add_argument('--scheduler_step_size', type=int, default=30,
        help='step size for scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.8,
        help='gamma for scheduler')
    parser.add_argument('--svd_ft_optim', type=str, default="adam", choices=['adam', 'adamw', 'sgd', 'sgdm'],
        help='svd ft optimizer')
    parser.add_argument('--svd_emb', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use svd on embedding layer')
    parser.add_argument('--use_S_gradinfo_init', action=argparse.BooleanOptionalAction, default=False,
        help='whether use S_gradinfo_init')
    parser.add_argument('--svd_layeridx', type=int, default=99, 
        help='svd layer index for local ft')
    parser.add_argument('--is_quant_aware_ft', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use quant aware ft')
    parser.add_argument('--is_per_head_svd', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use per head svd in QKV')
    parser.add_argument('--is_q_headnum', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use per q head svd in GQA')
    parser.add_argument('--misalign_text_image', action=argparse.BooleanOptionalAction, default=False, 
        help='whether misalign text and image for grad info')
    parser.add_argument('--is_taylor', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use taylor expansion to compute grad info')
    parser.add_argument('--taylor_order', type=int, default=2, 
        help='the taylor expansion order for S_grad_squared')
    parser.add_argument('--add_taylor_first', action=argparse.BooleanOptionalAction, default=False, 
        help='whether add taylor first order to grad info')
    parser.add_argument('--label_shift', action=argparse.BooleanOptionalAction, default=False,  
        help='shift llm label to the left')
    parser.add_argument('--is_rank_allocate_ft', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use rank allocate ft')
    parser.add_argument('--svd_modules', type=str, default=None, choices=[None,'qkv','kv','attn', 'all', 'mlp','gaup', 'all_sep','all_sep_mlpfine','all_sep_fine', 'down', 'o', 'all_sep_fine_wo_down', 'all_sep_rank_allocate_qkvonly', 'hybrid_ffn'],
        help='Apply SVD on which modules')
    parser.add_argument('--use_hybrid_ffn', action=argparse.BooleanOptionalAction, default=False,
        help='Replace FFN/MLP blocks with HybridSVDQuantFFN after regular LM SVD setup. '
             'This is independent from --svd_modules and can be combined with attention-only SVD such as qkv/attn/o.')
    parser.add_argument('--svd_ffn_ratio', type=float, default=0.2,
        help='Fraction of FFN hidden channels assigned to the SVD group in hybrid_ffn mode (rest go to quant)')
    parser.add_argument('--hybrid_ffn_rank_ratio', type=float, default=None,
        help='Rank ratio used only for the SVD subset inside hybrid_ffn. Defaults to --rank_ratio when unset.')
    parser.add_argument('--hybrid_real_int8', action=argparse.BooleanOptionalAction, default=False,
        help='Use real INT8 weight quantization (per-ch symmetric, weights stored as torch.int8) '
             'for the quant group and down_proj in hybrid_ffn mode instead of fake quant (GPTQ)')
    parser.add_argument('--hybrid_ffn_analyze_path', type=str, default=None,
        help='[Phase 1] Run offline channel-group analysis and save result to this JSON path. '
        'When unset together with --hybrid_ffn_groups_path, a default cache path under '
             'the shared calibration cache act_cache_dir/hybrid_ffn/ is used.')
    parser.add_argument('--hybrid_ffn_groups_path', type=str, default=None,
        help='[Phase 2] Load pre-computed channel groups from this JSON path (produced by '
             '--hybrid_ffn_analyze_path) and apply directly, skipping calibration. When unset '
        'together with --hybrid_ffn_analyze_path, the code auto-reuses the default cache '
             'under the shared calibration cache act_cache_dir/hybrid_ffn/ when available.')
    parser.add_argument('--hybrid_ffn_had', action=argparse.BooleanOptionalAction, default=False,
        help='Apply online Hadamard transform to the FFN hidden activations h=cat([h_svd, h_q]) '
             'before down_proj in hybrid_ffn mode.  Mirrors standard QuaRot online-Hadamard to '
             'reduce outliers in down_proj input quantization.  H^{-1} is pre-absorbed into '
             'W_down weights; runtime cost is one Hadamard per FFN layer forward pass.')
    parser.add_argument('--hybrid_ffn_svd_latent_rotate', action=argparse.BooleanOptionalAction, default=False,
        help='Insert an orthogonal reparameterization pair in each hybrid_ffn SVD latent space: '
             'B <- R B and A <- A R^T. FP output is unchanged, but W/A quantization sees rotated '
             'SVD factors and latent activations.')
    parser.add_argument('--hybrid_ffn_svd_latent_rotate_mode', type=str, default='hadamard',
        choices=['hadamard', 'random'],
        help='Orthogonal rotation used by --hybrid_ffn_svd_latent_rotate.')
    parser.add_argument('--hybrid_ffn_skip_layers', type=int, nargs='*', default=None,
        help='Layer indices whose FFN should NOT be replaced with HybridSVDQuantFFN. '
             'These layers keep their original (pure-quantization) FFN. '
             'Useful for skipping SVD-sensitive layers (e.g. the last layer). '
             'Example: --hybrid_ffn_skip_layers 31  or  --hybrid_ffn_skip_layers 28 29 30 31')
    parser.add_argument('--hybrid_ffn_fp16_quant_group', action=argparse.BooleanOptionalAction, default=False,
        help='Keep the non-SVD (quant) group channels of HybridSVDQuantFFN in FP16 '
             'instead of applying W8A8 quantization. Useful as an ablation to isolate '
             'the contribution of quantization vs. SVD compression.')
    parser.add_argument('--hybrid_ffn_act_aware', action=argparse.BooleanOptionalAction, default=False,
        help='Apply SVDLLM-style input whitening when constructing SVD factors for '
             'hybrid FFN SVD channels. Mode is controlled by --hybrid_ffn_act_aware_mode.')
    parser.add_argument('--hybrid_ffn_act_aware_mode', type=str, default='chol',
        choices=['chol', 'diag'],
        help='Whitening mode when --hybrid_ffn_act_aware is enabled.\n'
             '"chol": full lower-triangular Cholesky of X^T X (SVDLLM), W <- W @ L, '
             'requires L^{-1} to un-whiten V (can amplify B-matrix row norms).\n'
             '"diag": diagonal variance scaling, W <- W * sqrt(Var[X_j]), '
             'un-whitens via V / scale (no matrix inversion, numerically stable).')
    parser.add_argument('--svd_ffn_calib_method', type=str, default='error_compare',
        choices=['error_compare', 'loss_calibration', 'gradient_loss', 'svd_aware'],
        help='Channel selection method for hybrid_ffn. '
             '"error_compare": closed-form SVD/quant error comparison (fast, ~1 min). '
             '"loss_calibration": greedy FFN MSE, SVD-first, GPTQ quantization. '
             '"gradient_loss": SVD-first, uses final CE-loss gradient to select channels. '
             '"svd_aware": SVD-first greedy MSE selection using RTN quantization; '
             'moves channels to RTN quant only if RTN outperforms SVD on calib MSE.')
    parser.add_argument('--use_true_param_ratio', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use rank_ratio as param_ratio as did in asvd/svdllm for share svd')
    parser.add_argument('--use_all_true_param_ratio', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use all matrix true param ratio(just final /num_head)')
    parser.add_argument('--use_param_ratio', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use rank_ratio as param_ratio in gradinfo as did in asvd/svdllm')
    parser.add_argument('--use_group_svd', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use rank_ratio as param_ratio in GQA specifically, rank_Q = group_ratio * rank_KV')
    parser.add_argument('--group_ratio', type=float, default=0.0,
        help='group ratio for GQA specifically, rank_Q = group_ratio * rank_KV, also control grad multiplier for qkv')
    parser.add_argument('--rank_ratio', type=float, default=1.0,
        help='rank ratio for SVD compression, lowrank param = linear param * rank_ratio')
    parser.add_argument('--had_svd', action=argparse.BooleanOptionalAction, default=False, 
        help='had then svd')
    parser.add_argument('--had_rank', action=argparse.BooleanOptionalAction, default=False, 
        help='had on rank dimension')
    parser.add_argument('--svd_lm', action=argparse.BooleanOptionalAction, default=False,  
        help='Apply SVD compression to language model')
    parser.add_argument('--svd_lm_localft', action=argparse.BooleanOptionalAction, default=False,  
        help='Apply SVD ft to language model')
    parser.add_argument('--weighted_svd', action=argparse.BooleanOptionalAction, default=False,
        help='weighted svd using grad info')
    parser.add_argument('--weighted_none_svd_qat', action=argparse.BooleanOptionalAction, default=False,
        help='weighted qat using grad info for none svd linears')
    parser.add_argument('--qat_L_off', action=argparse.BooleanOptionalAction, default=False,
        help='disable Linear ft in qat (for linear layers without SVD compression)')
    parser.add_argument('--qat_uv_reg', action=argparse.BooleanOptionalAction, default=False,
        help='regularization for U/V in qat')
    parser.add_argument('--qat_uv_reg_scale', action=argparse.BooleanOptionalAction, default=False,
        help='scale regularization for U/V in qat')
    parser.add_argument('--qat_uv_reg_alpha', type=float, default=0.1,
        help='regularization alpha for U/V in qat')
    parser.add_argument('--svd_ft_mode', type=str, default="weight", choices=['output', 'weight'],
        help='local ft svd mode')
    parser.add_argument('--qat_start_iter', type=float, default=0.0,
        help='start iteration ratio for qat [0.0 - 1.0]')
    parser.add_argument('--qat_optim_R', action=argparse.BooleanOptionalAction, default=False,
        help='use R optimizer for qat')
    parser.add_argument('--qat_param_update_freq', type=int, default=1,
        help='frequency of parameter update for qat')
    parser.add_argument('--localft_iters', type=int, default=100,
        help='number of iterations for local ft')
    parser.add_argument('--localft_lr', type=float, default=1e-4,
        help='learning rate for local ft')
    parser.add_argument('--localft_lr_R', type=float, default=None,
        help='learning rate for QAT rotation matrix, deprecated')
    parser.add_argument('--qat_lr', type=float, default=None,
        help='learning rate for QAT rotation matrix, deprecated')
    parser.add_argument('--qat_lr_UV', type=float, default=None,
        help='learning rate for QAT UV matrix')
    parser.add_argument('--qat_lr_R', type=float, default=None,
        help='learning rate for QAT rotation matrix')
    parser.add_argument('--svd_vit',action=argparse.BooleanOptionalAction, default=False,  
        help='Apply SVD compression to vision model')
    parser.add_argument('--set_seedrot',action=argparse.BooleanOptionalAction, default=False,  
        help='whether apply same seed for rotation')
    parser.add_argument('--svd_mode', type=str, default="UV", 
        help='how to fold in Sigma into UV')
    parser.add_argument('--kv_fuse', action=argparse.BooleanOptionalAction, default=False, 
        help='use concat KV proj to have share V')
    parser.add_argument('--qkv_fuse', action=argparse.BooleanOptionalAction, default=False, 
        help='use concat QKV proj to have share V')
    parser.add_argument('--mlp_fuse', action=argparse.BooleanOptionalAction, default=False, 
        help='use concat up/gated to share Mlp up proj')
    parser.add_argument('--act_aware',action=argparse.BooleanOptionalAction, default=False, 
        help='use act aware SVD with calibration')
    parser.add_argument('--act_calib_fake_quant', action=argparse.BooleanOptionalAction, default=False,
        help='use fake-quantized activations only when collecting activation calibration statistics')
    parser.add_argument('--act_calib_debug', action=argparse.BooleanOptionalAction, default=False,
        help='Dump per-layer Gram/covariance, Cholesky L, inverse, and (in svd_utils) full SVD singular values S for act-aware cholesky path.')
    parser.add_argument('--act_calib_debug_dir', type=str, default=None,
        help='Output directory for act_calib_debug .pt files; defaults to save_path.')
    parser.add_argument('--act_calib_debug_empirical_fisher', action=argparse.BooleanOptionalAction, default=False,
        help='Capture projected per-head empirical Fisher sketches for selected q/k/v modules during loss-aware grad calibration.')
    parser.add_argument('--act_calib_debug_empirical_fisher_layers', type=str, default=None,
        help='Comma-separated layer ids or ranges for empirical Fisher debug capture. Defaults to an evenly spaced subset.')
    parser.add_argument('--act_calib_debug_empirical_fisher_num_layers', type=int, default=8,
        help='Number of layers to sample for empirical Fisher debug capture when explicit layers are not provided.')
    parser.add_argument('--act_calib_debug_empirical_fisher_heads', type=str, default=None,
        help='Comma-separated head ids or ranges for empirical Fisher debug capture. Defaults to an evenly spaced subset per module.')
    parser.add_argument('--act_calib_debug_empirical_fisher_num_heads', type=int, default=8,
        help='Number of heads to sample per selected layer/proj for empirical Fisher debug capture when explicit heads are not provided.')
    parser.add_argument('--act_calib_debug_empirical_fisher_input_proj_dim', type=int, default=8,
        help='Input projection dimension used by projected empirical Fisher debug capture.')
    parser.add_argument('--act_calib_debug_empirical_fisher_output_proj_dim', type=int, default=8,
        help='Output projection dimension used by projected empirical Fisher debug capture.')
    parser.add_argument('--act_calib_debug_empirical_fisher_seed', type=int, default=0,
        help='Seed used to build deterministic projection sketches for empirical Fisher debug capture.')
    parser.add_argument('--act_alpha', type=float, default=0.5,
        help='sensitivity of ASVD S construction')
    parser.add_argument('--loss_aware', action=argparse.BooleanOptionalAction, default=False,
        help='calculate loss aware SVD with calibration')
    parser.add_argument('--reasoning_calib', action=argparse.BooleanOptionalAction, default=False,
        help='Use reasoning-aware calibration traces for activation/loss-aware SVD statistics. Default keeps legacy calibration.')
    parser.add_argument('--reasoning_calib_mode', type=str, default='rollout_teacher_forcing',
        choices=['rollout_teacher_forcing'],
        help='Reasoning calibration mode. rollout_teacher_forcing generates traces, then replays them with teacher-forced loss.')
    parser.add_argument('--reasoning_max_new_tokens', type=int, default=4096,
        help='Maximum new tokens when generating reasoning calibration traces.')
    parser.add_argument('--reasoning_temperature', type=float, default=0.0,
        help='Sampling temperature for reasoning trace generation; 0 uses deterministic generation.')
    parser.add_argument('--reasoning_rollout_engine', type=str, default='vllm',
        choices=['vllm', 'hf'],
        help='Engine used to produce reasoning trace caches. vllm expects an offline cache; hf runs legacy online generation.')
    parser.add_argument('--reasoning_weight', type=float, default=1.0,
        help='Loss weight for reasoning/thinking tokens during reasoning calibration.')
    parser.add_argument('--answer_weight', type=float, default=1.0,
        help='Loss weight for final-answer tokens during reasoning calibration.')
    parser.add_argument('--reasoning_trace_cache_dir', type=str, default=None,
        help='Directory for generated reasoning trace caches. Defaults to act_cache_dir/cache.')
    parser.add_argument('--loss_aware_and_truncate', action=argparse.BooleanOptionalAction, default=False,
        help='calculate loss aware SVD with calibration and truncate')
    parser.add_argument('--truncate_min_rank_per_head', type=int, default=1,
        help='Minimum selected rank per head for per-head loss-aware truncate. <=0 disables the floor.')
    parser.add_argument('--truncate_min_rank_modules', type=str, default='qkv', choices=['q', 'k', 'v', 'qv', 'qkv'],
        help='Which module groups receive per-head rank floor during loss_aware_and_truncate.')
    parser.add_argument('--truncate_min_rank_scope', type=str, default='all_layers', choices=['all_layers', 'front_layers', 'middle_layers', 'late_layers', 'custom'],
        help='Layer scope for per-head rank floor during loss_aware_and_truncate.')
    parser.add_argument('--truncate_min_rank_layers', type=str, default=None,
        help='Comma-separated layer ids or ranges (e.g. 0,1,3-5) used when truncate_min_rank_scope=custom.')
    parser.add_argument('--truncate_min_rank_strict', action=argparse.BooleanOptionalAction, default=True,
        help='Keep per-module total selected budget fixed when applying per-head rank floor by rebalancing across heads.')
    parser.add_argument('--truncate_balance_penalty', action=argparse.BooleanOptionalAction, default=False,
        help='Use a soft head-balance penalty during per-head loss-aware truncate instead of a hard minimum-rank floor.')
    parser.add_argument('--truncate_balance_lambda', type=float, default=0.0,
        help='Strength of the soft head-balance penalty used when --truncate_balance_penalty is enabled.')
    parser.add_argument('--truncate_hybrid_grad_layer_budget', action=argparse.BooleanOptionalAction, default=False,
        help='Use grad-info to set per-layer QKV budgets, while keeping newRA scores for within-layer rank selection.')
    parser.add_argument('--truncate_hybrid_loss_aware_layer_budget', action=argparse.BooleanOptionalAction, default=False,
        help='Use loss-aware per-layer singular-value energy to set per-layer QKV budgets, while keeping loss-aware scores for within-layer rank selection.')
    parser.add_argument('--truncate_hybrid_modules', type=str, default=None, choices=['q', 'k', 'v', 'qv', 'qkv'],
        help='Which q/k/v modules receive hybrid layer-budget guidance. Defaults to qkv when unset.')
    parser.add_argument('--truncate_mixed_qkv_budget', action=argparse.BooleanOptionalAction, default=False,
        help='Use a shared q/k/v top-k pool instead of separate per-module budgets when q_proj/k_proj/v_proj are selected together.')
    parser.add_argument('--calib_method', type=str, default='abs_mean', choices=['abs_mean', 'abs_max', 'cholesky'], 
        help='calibation method for act aware SVD')
    parser.add_argument('--fisher_info', action=argparse.BooleanOptionalAction, default=False,
        help='calculate Fisher Information for rank allocations for each layer')
    parser.add_argument('--grad_info', action=argparse.BooleanOptionalAction, default=False,
        help='calculate Grad Information for rank allocations for each layer')
    parser.add_argument('--use_cache', type=lambda x: x.lower() in ['true', '1'], default=True, 
                        help='Use previous cached calibration results, default is True!')
    parser.add_argument('--progressive_svd', action=argparse.BooleanOptionalAction, default=False, 
        help='use progressive svd to allocate rank for each layer, use with grad_info, currently only support svd_lm with qkv_fuse')
    parser.add_argument('--pre_svd_rotate', action=argparse.BooleanOptionalAction, default=False, 
        help='whether rotate the weights before svd')
    parser.add_argument('--profile_method', action=argparse.BooleanOptionalAction, default=False, 
        help='whether to skip the eval part, just to profile the model using calibrated dataset')
    parser.add_argument('--beta_then_svd', action=argparse.BooleanOptionalAction, default=False, 
        help='lm svd depend on vit output, so this decides whether svd happens before or after vit module')
    parser.add_argument('--vit_module', action=argparse.BooleanOptionalAction, default=False, 
        help='whether we quantize/rotate vit model')
    parser.add_argument('--vit_online', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use online rotation vit model')
    parser.add_argument('--vit_mmoff', action=argparse.BooleanOptionalAction, default=False, 
        help='whether skip quantize  mmprojector')
    parser.add_argument('--mm_rh', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use rh after gelu in mmprojector')
    parser.add_argument('--lm_off', action=argparse.BooleanOptionalAction, default=False, 
        help='whether skip language model rotate/quantize')
    parser.add_argument('--cache_in_log', action=argparse.BooleanOptionalAction, default=False, 
        help='whether cache calibration results in log file')
    parser.add_argument('--cache_file', type=str, default=None,
        help='path to cached calibration results, default is None!')
    parser.add_argument('--calib_cache_dir', type=str, default=None,
        help='directory of released calibration caches for this model (act_input.pt, output_grad.pt, '
             'grad_info.pt, grad_scores.pt and calib_cache.json; see utils/calib_cache.py). They are read '
             'instead of the shared cache and never recomputed or written.')
    parser.add_argument(
        '--act_cache_namespace',
        type=str,
        default='shared_cache',
        help=(
            'Single relative directory name used for activation/Fisher caches '
            'under experiments/<model>. Use distinct names for protocol-locked studies.'
        ),
    )
    parser.add_argument('--svd_info_cache_dir', type=str, default=None,
        help='directory containing precomputed local-FT SVD structure for checkpoint-only evaluation')
    parser.add_argument('--grad_alpha', type=float, default=1.0,
        help='sensitivity of grad score construction')
    parser.add_argument('--bs_to_nsamples', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use full sample size in llava v1.6 beta learning')
    parser.add_argument('--token_length', type=int, default=-1, 
        help='truncate grad info input token length')
    parser.add_argument('--label_mode', type=str, default="q-a", 
        help='how to set input and label for loss and grad')
    parser.add_argument('--case_study', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use case study for llava evaluation')
    
    # Save/Load Quantized Model Arguments
    parser.add_argument('--load_qmodel_path', type=str, default=None,
                        help='Load the quantized model from the specified path!')
    parser.add_argument('--save_qmodel_path', type=str, default=None, 
                        help='Save the quantized model to the specified path!')
    parser.add_argument(
        '--export_fused_qwen',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='If set, run SVD/quant pipeline then fuse SVDLinear into nn.Linear and save checkpoint instead of evaluation.',
    )
    parser.add_argument(
        '--export_dir',
        type=str,
        default=None,
        help='Output directory for fused model when --export_fused_qwen is set. Defaults to [save_path]/fused_model.',
    )

    # WandB Arguments
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)


    #Experiments Arguments
    parser.add_argument('--setting', type=str, default="", help='The path to save experiment data, '
                                                                    'including quantized models, dumped layer inputs, etc. The data will be saved in experiments/[model]/save_name. Default: [datetime].')
    parser.add_argument('--save_name', type=str, default=None, help='The path to save experiment data, '
                                                                    'including quantized models, dumped layer inputs, etc. The data will be saved in experiments/[model]/save_name. Default: [datetime].')
    parser.add_argument('--capture_layer_io', action=argparse.BooleanOptionalAction, default=False,
                        help='Capture the input and output of the specified decoder layer and dump into a file')
    parser.add_argument('--layer_idx', type=int, default=10, help='Which decoder layer to capture')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False,
                        help='whether to print the response of the model')
    # LM Eval Arguments
    parser.add_argument("--lm_eval", action="store_true", help="Evaluate the model on LM Eval tasks.")
    parser.add_argument(
        '--tasks',
        type=lambda s: s.split(','),
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada", "boolq"],
    )
    parser.add_argument(
        '--vlmtasks',
        nargs='+',
        default=["ScienceQA_TEST"],
    )
    parser.add_argument('--lm_eval_batch_size', type=int, default=16, help='Batch size for evaluating with lm eval harness.')
    parser.add_argument(
        "--distribute",
        action="store_true",
        help="Distribute the model on multiple GPUs for evaluation.",
    )

    args = parser.parse_args()
    args = normalize_hybrid_ffn_args(args)
    if args.localft_lr_R is None:
        args.localft_lr_R = args.localft_lr
    if args.save_name is None:
        args.save_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.setting = f'W{args.w_bits}A{args.a_bits}K{args.k_bits}V{args.v_bits}'+args.setting
    setattr(args, 'save_path',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', args.model, args.setting))
    os.makedirs(args.save_path, exist_ok=True)
    cache_namespace = str(args.act_cache_namespace).strip()
    if (
        not cache_namespace
        or os.path.isabs(cache_namespace)
        or cache_namespace in {'.', '..'}
        or os.path.basename(cache_namespace) != cache_namespace
        or cache_namespace.startswith('.')
    ):
        raise ValueError(
            "--act_cache_namespace must be one visible relative directory name; "
            f"got {args.act_cache_namespace!r}."
        )
    args.act_cache_dir = cache_namespace
    setattr(args, 'act_cache_dir',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', args.model, args.act_cache_dir))
    config_logging(os.path.join(args.save_path, f'{args.save_name}.log'))
    if args.load_qmodel_path:
        qmodel_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'experiments',
            args.model,
            'checkpoint',
        )
        if not os.path.isabs(args.load_qmodel_path):
            load_name = args.load_qmodel_path
            if not load_name.endswith('.pth'):
                load_name = f'{load_name}.pth'
            args.load_qmodel_path = os.path.join(qmodel_dir, load_name)
    if args.save_qmodel_path:
        qmodel_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'experiments',
            args.model,
            'checkpoint',
        )
        if os.path.isabs(args.save_qmodel_path):
            save_path = args.save_qmodel_path
        else:
            save_path = os.path.join(qmodel_dir, args.save_qmodel_path)
        if not save_path.endswith('.pth'):
            save_path = f'{save_path}.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        args.save_qmodel_path = save_path
    
    assert args.a_groupsize == args.w_groupsize, 'a_groupsize should be the same as w_groupsize!'
    assert args.k_pre_rope == False, 'Pre-RoPE quantization is not supported yet!'

    if args.model == 'facebook/opt-125m' or args.model == 'facebook/opt-1.3b':
        logging.warning('Warning: OPT-125M/1.3B is only for debugging purposes!!')


    if args.wandb:
        assert args.wandb_id is not None and args.wandb_project is not None, 'WandB ID/project is not provided!'
    if args.calib_cache_dir:
        from . import calib_cache
        calib_cache.check_args(args)
        
    logging.info('Arguments: ')
    logging.info(pprint.pformat(vars(args)))
    logging.info('--' * 30)
    if args.lm_eval:
        from lm_eval import tasks
        from lm_eval import utils as lm_eval_utils
        from lm_eval.tasks import initialize_tasks
        initialize_tasks()
        for task in args.tasks:
            if task not in lm_eval_utils.MultiChoice(tasks.ALL_TASKS):
                raise ValueError(f"Invalid task: {task}")
    return args


def normalize_hybrid_ffn_args(args):
    """Normalize hybrid FFN flags and validate unsupported combinations.

    Backward compatibility:
      --svd_modules hybrid_ffn  ->  --use_hybrid_ffn with no regular LM SVD modules.
    """
    if getattr(args, 'svd_modules', None) == 'hybrid_ffn':
        args.use_hybrid_ffn = True
        args.svd_modules = None

    if not getattr(args, 'use_hybrid_ffn', False):
        return args

    incompatible_svd_modules = {
        'all',
        'mlp',
        'gaup',
        'down',
        'all_sep',
        'all_sep_mlpfine',
        'all_sep_fine',
        'all_sep_fine_wo_down',
        'all_sep_rank_allocate_qkvonly',
    }
    if getattr(args, 'svd_modules', None) in incompatible_svd_modules:
        raise ValueError(
            "--use_hybrid_ffn is only compatible with attention-only LM SVD selections "
            "(None/qkv/kv/attn/o). It cannot be combined with FFN-including svd_modules "
            f"such as {args.svd_modules!r}."
        )

    return args


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )

def distribute_model(model) -> None:
    """Distribute the model across available GPUs. NB: only implemented for Llama-2."""
    no_split_module_classes = ['LlamaDecoderLayer']
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )

    cleanup_memory()
