# ⚡ LSVD kernels

Triton kernels for LSVD's low-precision low-rank layer and the layer-wise decoding-latency benchmark of the paper
(Sec. 3.6 / Fig. 5). They are independent of the accuracy code in `fake_quant/`, which uses fake quantization.

## 📦 Contents

| Path | Content |
|---|---|
| `layer_bench/fd_kernel/` | W8A8 low-rank flash-decoding attention: per-head int8 latent KV cache, keys reconstructed from the latents with RoPE inside the kernel; the new token's latent K and V are quantized into the cache by a fused kernel (adapted from LightLLM's flash-decoding and KV-cache kernels) |
| `layer_bench/ffn_kernel/mix_ffn_kernel.py` | LSVD hybrid low-rank FFN (`mixffn`): fused SVD up-projection and dequantization, SwiGLU from the latents, dense W8A8 gate/up for the RTN channels, RMSNorm with per-token quantization, SVD and dense down-projection |
| `layer_bench/ffn_kernel/baseline_kernel.py` | the LSVD attention block and the FP16 reference layer (eager attention, Flash Decoding with PyTorch SDPA; the new token's query and key get RoPE, and its key and value are written into a preallocated KV cache in place) |
| `layer_bench/ffn_kernel/utils.py` | autotuning configurations, autotune-cache handling, timing helpers |
| `layer_bench/run_autotune_triton_int8_layer.py`, `layer_bench/benchmark_triton_int8_layer.py` | autotuning and the benchmark harness |
| `run_fig5.sh` | the Fig. 5 benchmark |

## 🔧 Installation

The kernels need only PyTorch and Triton, on an NVIDIA GPU with int8 tensor cores (sm_80 or newer). A separate
environment is recommended:

```bash
conda create -n lsvd_kernels python=3.10 -y && conda activate lsvd_kernels
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126   # includes triton 3.2.0
pip install numpy pyyaml matplotlib
```

Blackwell GPUs (e.g. RTX 5090) need a PyTorch build with CUDA 12.8 or newer.

## ▶️ Decoding latency (Fig. 5)

```bash
bash kernels/run_fig5.sh
```

The script autotunes the LSVD kernels for the current GPU (cached in `kernels/autotune_cache/`; reuse with
`SKIP_AUTOTUNE=1`) and then times one transformer layer with CUDA-graph replay for Eager attention, Flash Decoding
and LSVD. It prints the per-layer, attention and FFN latency for each method and context length, and writes the logs
and a speedup plot to `results/fig5_<GPU>/`.
Latency depends on the GPU; the paper reports an RTX 4090 and an RTX 5090.

| Setting | Default (environment variable) |
|---|---|
| batch, decoding step | 32 (`BATCH_SIZE`), one query token |
| context lengths | 4096, 6144, 8192 (`CONTEXT_LENS`) |
| layer shape | hidden 4096, FFN 14336 (`INTERMEDIATE_SIZE`), 32 heads of dimension 128 |
| LSVD | attention rho_1 = 0.5 (`LSVD_RHO1`); FFN: 50% of the channels in the SVD branch (`FFN_SVD_RATIO`), rank ratio 0.9 (`FFN_RANK_RATIO`) |
| timing | 300 repetitions (`REPEAT_TIME`) |
