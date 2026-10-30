import logging

from typing import Callable, Dict, Optional, Tuple

import torch

from torch import nn

import torch.nn.functional as F

import triton

import triton.language as tl

from .utils import (
    get_dense_down_dequant_autotune_config,
    get_dequant_autotune_config,
    get_dual_quantize_vec_autotune_config,
    get_dual_svd_up_a_dequant_autotune_config,
    get_mixed_gateup_swiglu_from_latents_autotune_config,
    get_rmsnorm_quant_autotune_config,
    get_svd_down_b_dequant_autotune_config,
    get_svd_rank_quant_autotune_config,
    get_swiglu_a2_vec_autotune_config,
    get_swiglu_quant_autotune_config,
    get_w8a8_dot_down_matmul_autotune_config,
    get_w8a8_dot_up_matmul_autotune_config,
    get_w8a8_svd_down_a_matmul_autotune_config,
    get_w8a8_svd_down_b_matmul_autotune_config,
    get_w8a8_svd_up_a_matmul_autotune_config,
    get_w8a8_svd_up_b_matmul_autotune_config,
)

_SCALE_DTYPE = torch.float16

def _maxq(bits: int) -> int:
    if bits == 8:
        return 127
    if bits == 4:
        return 7
    raise ValueError(f"Unsupported bit-width: {bits}")

def _minq(bits: int) -> int:
    return -(_maxq(bits) + 1)

def bench_stage_callable(fn, repeat_time: int = 100, bench_mode: str = "eager") -> float:
    quantiles = [0.5, 0.2, 0.8]
    if bench_mode != "cuda_graph":
        ms, _, _ = triton.testing.do_bench(fn, quantiles=quantiles, rep=repeat_time)
        return float(ms)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    ms, _, _ = triton.testing.do_bench(lambda: graph.replay(), quantiles=quantiles, rep=repeat_time)
    return float(ms)

@triton.jit
def _silu_mul_kernel(
    up_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = up * (gate * tl.sigmoid(gate))
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_silu_mul(up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if up.shape != gate.shape:
        raise ValueError(f"up/gate shape mismatch: {tuple(up.shape)} != {tuple(gate.shape)}")
    if not up.is_cuda:
        return up * F.silu(gate)
    if not up.is_contiguous() or not gate.is_contiguous():
        up = up.contiguous()
        gate = gate.contiguous()
    out = torch.empty_like(up)
    block_size = 1024
    grid = (triton.cdiv(up.numel(), block_size),)
    _silu_mul_kernel[grid](up, gate, out, up.numel(), BLOCK_SIZE=block_size)
    return out

@triton.autotune(
    configs=get_rmsnorm_quant_autotune_config(),
    key=["M", "K", "MAXQ"],
)
@triton.jit
def _rmsnorm_quant_kernel(
    x_ptr,
    norm_w_ptr,
    q_ptr,
    scale_ptr,
    M,
    K,
    eps,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_qm,
    stride_qk,
    MAXQ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = offs_m < M

    row_sumsq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = row_mask[:, None] & (offs_k[None, :] < K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        row_sumsq += tl.sum(x * x, axis=1)

    inv_rms = tl.rsqrt(row_sumsq / K + eps)

    row_max = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = row_mask[:, None] & (offs_k[None, :] < K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(norm_w_ptr + offs_k * stride_wk, mask=offs_k < K, other=0.0).to(tl.float32)
        normed = x * inv_rms[:, None] * w[None, :]
        row_max = tl.maximum(row_max, tl.max(tl.abs(normed), axis=1))

    scale = tl.maximum(row_max / MAXQ, 1e-8)
    tl.store(scale_ptr + offs_m, scale, mask=row_mask)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = row_mask[:, None] & (offs_k[None, :] < K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(norm_w_ptr + offs_k * stride_wk, mask=offs_k < K, other=0.0).to(tl.float32)
        normed = x * inv_rms[:, None] * w[None, :]
        scaled = normed / scale[:, None]
        rounded = tl.where(scaled >= 0, scaled + 0.5, scaled - 0.5).to(tl.int32)
        clamped = tl.maximum(tl.minimum(rounded, MAXQ), -(MAXQ + 1))
        tl.store(q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk, clamped.to(tl.int8), mask=mask)

@triton.autotune(
    configs=get_dequant_autotune_config(),
    key=["M", "N"],
)
@triton.jit
def _dequant_only_kernel(
    acc_ptr,
    out_ptr,
    x_scale_ptr,
    w_scale_ptr,
    M,
    N,
    stride_am,
    stride_an,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    acc = tl.load(acc_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, mask=mask, other=0).to(tl.float32)
    x_scale = tl.load(x_scale_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)[:, None]
    w_scale = tl.load(w_scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[None, :]
    out = acc * x_scale * w_scale
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, out.to(tl.float16), mask=mask)

def quantize_weights_per_channel(
    w: torch.Tensor,
    bits: int,
    scale_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    maxq = _maxq(bits)
    minq = _minq(bits)
    scale_dtype = scale_dtype or _SCALE_DTYPE
    scale = torch.maximum(w.abs().amax(dim=1) / maxq, torch.tensor(1e-8, device=w.device, dtype=w.dtype))
    q = torch.clamp(torch.round(w / scale.unsqueeze(-1)), minq, maxq).to(torch.int8)
    return q.contiguous(), scale.to(scale_dtype).contiguous()

def rmsnorm_quantize_per_token(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    bits: int,
    scale_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = x_2d.shape
    q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    scale_dtype = scale_dtype or _SCALE_DTYPE
    scale = torch.empty((m,), device=x.device, dtype=scale_dtype)
    grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]),)
    _rmsnorm_quant_kernel[grid](
        x_2d,
        norm_weight,
        q,
        scale,
        m,
        k,
        eps,
        x_2d.stride(0),
        x_2d.stride(1),
        norm_weight.stride(0),
        q.stride(0),
        q.stride(1),
        MAXQ=_maxq(bits),
    )
    return q, scale

def dequantize_only(
    acc: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    m, n = acc.shape
    out = torch.empty((m, n), device=acc.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),)
    _dequant_only_kernel[grid](
        acc,
        out,
        x_scale,
        w_scale,
        m,
        n,
        acc.stride(0),
        acc.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.norm = LlamaRMSNorm(hidden_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x = triton_silu_mul(self.up_proj(x), self.gate_proj(x))
        x = self.down_proj(x)
        return x + res

def make_base_ffn_inputs(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> Tuple[torch.Tensor, SwiGLUFFN]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.randn((batch_size, seq_len, hidden_size), device=device, dtype=dtype, generator=generator)
    module = SwiGLUFFN(hidden_size, intermediate_size).to(device=device, dtype=dtype)
    with torch.no_grad():
        module.norm.weight.copy_(torch.randn(module.norm.weight.shape, device=device, dtype=dtype, generator=generator))
        module.up_proj.weight.copy_(
            torch.randn(module.up_proj.weight.shape, device=device, dtype=dtype, generator=generator) / (hidden_size**0.5)
        )
        module.gate_proj.weight.copy_(
            torch.randn(module.gate_proj.weight.shape, device=device, dtype=dtype, generator=generator) / (hidden_size**0.5)
        )
        module.down_proj.weight.copy_(
            torch.randn(module.down_proj.weight.shape, device=device, dtype=dtype, generator=generator) / (intermediate_size**0.5)
        )
    return x.contiguous(), module

@triton.autotune(
    configs=get_swiglu_a2_vec_autotune_config(),
    key=["N", "MAXQ"],
)
@triton.jit
def _swiglu_a2_max_vec_kernel(
    up_acc_ptr,
    gate_acc_ptr,
    out_ptr,
    scale_ptr,
    x_scale_ptr,
    up_scale_ptr,
    gate_scale_ptr,
    M,
    N,
    stride_um,
    stride_un,
    stride_gm,
    stride_gn,
    stride_om,
    stride_on,
    MAXQ: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(axis=0)
    if row >= M:
        return

    row_max = tl.zeros((), dtype=tl.float32)
    for n_start in range(0, tl.cdiv(N, BLOCK_N)):
        offs_n = n_start * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        up_acc = tl.load(up_acc_ptr + row * stride_um + offs_n * stride_un, mask=mask, other=0).to(tl.float32)
        gate_acc = tl.load(gate_acc_ptr + row * stride_gm + offs_n * stride_gn, mask=mask, other=0).to(tl.float32)
        x_scale = tl.load(x_scale_ptr + row).to(tl.float32)
        up_scale = tl.load(up_scale_ptr + offs_n, mask=mask, other=0.0).to(tl.float32)
        gate_scale = tl.load(gate_scale_ptr + offs_n, mask=mask, other=0.0).to(tl.float32)
        up = up_acc * x_scale * up_scale
        gate = gate_acc * x_scale * gate_scale
        out = up * (gate * tl.sigmoid(gate))
        row_max = tl.maximum(row_max, tl.max(tl.abs(out), axis=0))

    scale = tl.maximum(row_max / MAXQ, 1e-8)
    tl.store(scale_ptr + row, scale)

    for n_start in range(0, tl.cdiv(N, BLOCK_N)):
        offs_n = n_start * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        up_acc = tl.load(up_acc_ptr + row * stride_um + offs_n * stride_un, mask=mask, other=0).to(tl.float32)
        gate_acc = tl.load(gate_acc_ptr + row * stride_gm + offs_n * stride_gn, mask=mask, other=0).to(tl.float32)
        x_scale = tl.load(x_scale_ptr + row).to(tl.float32)
        up_scale = tl.load(up_scale_ptr + offs_n, mask=mask, other=0.0).to(tl.float32)
        gate_scale = tl.load(gate_scale_ptr + offs_n, mask=mask, other=0.0).to(tl.float32)
        up = up_acc * x_scale * up_scale
        gate = gate_acc * x_scale * gate_scale
        out = up * (gate * tl.sigmoid(gate))
        tl.store(out_ptr + row * stride_om + offs_n * stride_on, out.to(tl.float16), mask=mask)

def swiglu_a2_compute_max(
    up_acc: torch.Tensor,
    gate_acc: torch.Tensor,
    x_scale: torch.Tensor,
    up_scale: torch.Tensor,
    gate_scale: torch.Tensor,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = up_acc.shape
    out = torch.empty((m, n), device=up_acc.device, dtype=torch.float16)
    scale = torch.empty((m,), device=up_acc.device, dtype=torch.float32)
    grid = (m,)
    _swiglu_a2_max_vec_kernel[grid](
        up_acc,
        gate_acc,
        out,
        scale,
        x_scale,
        up_scale,
        gate_scale,
        m,
        n,
        up_acc.stride(0),
        up_acc.stride(1),
        gate_acc.stride(0),
        gate_acc.stride(1),
        out.stride(0),
        out.stride(1),
        MAXQ=_maxq(bits),
    )
    return out, scale

@triton.autotune(configs=get_w8a8_dot_up_matmul_autotune_config(), key=["M", "N", "K"])
@triton.jit
def _w8a8_dot_up_matmul_kernel(
    a_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0,
        ).to(tl.int8)
        w = tl.load(
            w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=mask_k[:, None] & (offs_n[None, :] < N),
            other=0,
        ).to(tl.int8)
        acc += tl.dot(a, w)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

@triton.autotune(configs=get_w8a8_dot_down_matmul_autotune_config(), key=["M", "N", "K"])
@triton.jit
def _w8a8_dot_down_matmul_kernel(
    a_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0,
        ).to(tl.int8)
        w = tl.load(
            w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=mask_k[:, None] & (offs_n[None, :] < N),
            other=0,
        ).to(tl.int8)
        acc += tl.dot(a, w)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

def _w8a8_matmul(x_q: torch.Tensor, w_q: torch.Tensor, kernel) -> torch.Tensor:
    x_q = x_q.contiguous()
    w_q = w_q.contiguous()
    m = x_q.shape[0]
    k = x_q.shape[1]
    n = w_q.shape[0]
    out = torch.empty((m, n), device=x_q.device, dtype=torch.int32)
    grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]), triton.cdiv(n, META["BLOCK_N"]))
    kernel[grid](
        x_q,
        w_q,
        out,
        m,
        n,
        k,
        x_q.stride(0),
        x_q.stride(1),
        w_q.stride(0),
        w_q.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out

def w8a8_dot_matmul(x_q: torch.Tensor, w_q: torch.Tensor) -> torch.Tensor:
    return _w8a8_matmul(x_q, w_q, _w8a8_dot_up_matmul_kernel)

def w8a8_dot_down_matmul(x_q: torch.Tensor, w_q: torch.Tensor) -> torch.Tensor:
    return _w8a8_matmul(x_q, w_q, _w8a8_dot_down_matmul_kernel)

DEFAULT_RANK_RATIO = 0.5

DEFAULT_ROUND_TO = 64

def _rounded_rank(in_features: int, out_features: int, rank_ratio: float, round_to: int) -> int:
    if rank_ratio <= 0.0:
        return 0
    rank = int(in_features * out_features * rank_ratio / (in_features + out_features))
    rank = max(1, min(rank, in_features, out_features))
    if round_to > 1 and rank >= round_to:
        rank = max(round_to, (rank // round_to) * round_to)
    return max(1, min(rank, in_features, out_features))

def _init_low_rank_factors(weight: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if rank <= 0:
        raise ValueError("rank must be positive")
    out_features, in_features = weight.shape
    # v4 is a synthetic low-rank branch for kernel experiments: keep the same
    # rank formula, but initialize the factors directly instead of spending
    # time on an actual SVD decomposition of the dense weight.
    a = torch.randn((rank, in_features), device=weight.device, dtype=weight.dtype) / (in_features**0.5)
    b = torch.randn((out_features, rank), device=weight.device, dtype=weight.dtype) / (rank**0.5)
    return a.contiguous(), b.contiguous()

@triton.jit
def _w8a8_svd_dot_matmul_kernel(
    a_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0,
        ).to(tl.int8)
        w = tl.load(
            w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=mask_k[:, None] & (offs_n[None, :] < N),
            other=0,
        ).to(tl.int8)
        acc += tl.dot(a, w)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

@triton.autotune(configs=get_w8a8_svd_up_a_matmul_autotune_config(), key=["M", "N", "K"])
@triton.jit
def _w8a8_svd_up_a_matmul_kernel(
    a_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    _w8a8_svd_dot_matmul_kernel(
        a_ptr,
        w_ptr,
        out_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

@triton.autotune(configs=get_w8a8_svd_up_b_matmul_autotune_config(), key=["M", "N", "K"])
@triton.jit
def _w8a8_svd_up_b_matmul_kernel(
    a_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    _w8a8_svd_dot_matmul_kernel(
        a_ptr,
        w_ptr,
        out_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

@triton.autotune(configs=get_w8a8_svd_down_a_matmul_autotune_config(), key=["M", "N", "K"])
@triton.jit
def _w8a8_svd_down_a_matmul_kernel(
    a_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    _w8a8_svd_dot_matmul_kernel(
        a_ptr,
        w_ptr,
        out_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

@triton.autotune(configs=get_w8a8_svd_down_b_matmul_autotune_config(), key=["M", "N", "K"])
@triton.jit
def _w8a8_svd_down_b_matmul_kernel(
    a_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    _w8a8_svd_dot_matmul_kernel(
        a_ptr,
        w_ptr,
        out_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

@triton.autotune(configs=get_svd_rank_quant_autotune_config(), key=["R", "MAXQ"])
@triton.jit
def _svd_rank_dequant_quant_up_kernel(
    acc_ptr,
    q_ptr,
    scale_ptr,
    x_scale_ptr,
    w_scale_ptr,
    M,
    R,
    stride_am,
    stride_ar,
    stride_qm,
    stride_qr,
    MAXQ: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    row = tl.program_id(axis=0)
    if row >= M:
        return

    row_max = tl.zeros((), dtype=tl.float32)
    x_scale = tl.load(x_scale_ptr + row).to(tl.float32)
    for r_start in range(0, tl.cdiv(R, BLOCK_R)):
        offs_r = r_start * BLOCK_R + tl.arange(0, BLOCK_R)
        mask = offs_r < R
        acc = tl.load(acc_ptr + row * stride_am + offs_r * stride_ar, mask=mask, other=0).to(tl.float32)
        w_scale = tl.load(w_scale_ptr + offs_r, mask=mask, other=0.0).to(tl.float32)
        val = acc * x_scale * w_scale
        row_max = tl.maximum(row_max, tl.max(tl.abs(val), axis=0))

    scale = tl.maximum(row_max / MAXQ, 1e-8)
    tl.store(scale_ptr + row, scale)

    for r_start in range(0, tl.cdiv(R, BLOCK_R)):
        offs_r = r_start * BLOCK_R + tl.arange(0, BLOCK_R)
        mask = offs_r < R
        acc = tl.load(acc_ptr + row * stride_am + offs_r * stride_ar, mask=mask, other=0).to(tl.float32)
        w_scale = tl.load(w_scale_ptr + offs_r, mask=mask, other=0.0).to(tl.float32)
        val = acc * x_scale * w_scale
        scaled = val / scale
        rounded = tl.where(scaled >= 0, scaled + 0.5, scaled - 0.5).to(tl.int32)
        clamped = tl.maximum(tl.minimum(rounded, MAXQ), -(MAXQ + 1))
        tl.store(q_ptr + row * stride_qm + offs_r * stride_qr, clamped.to(tl.int8), mask=mask)

@triton.autotune(configs=get_svd_rank_quant_autotune_config(), key=["R", "MAXQ"])
@triton.jit
def _svd_rank_dequant_quant_down_kernel(
    acc_ptr,
    q_ptr,
    scale_ptr,
    x_scale_ptr,
    w_scale_ptr,
    M,
    R,
    stride_am,
    stride_ar,
    stride_qm,
    stride_qr,
    MAXQ: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    row = tl.program_id(axis=0)
    if row >= M:
        return

    row_max = tl.zeros((), dtype=tl.float32)
    x_scale = tl.load(x_scale_ptr + row).to(tl.float32)
    for r_start in range(0, tl.cdiv(R, BLOCK_R)):
        offs_r = r_start * BLOCK_R + tl.arange(0, BLOCK_R)
        mask = offs_r < R
        acc = tl.load(acc_ptr + row * stride_am + offs_r * stride_ar, mask=mask, other=0).to(tl.float32)
        w_scale = tl.load(w_scale_ptr + offs_r, mask=mask, other=0.0).to(tl.float32)
        val = acc * x_scale * w_scale
        row_max = tl.maximum(row_max, tl.max(tl.abs(val), axis=0))

    scale = tl.maximum(row_max / MAXQ, 1e-8)
    tl.store(scale_ptr + row, scale)

    for r_start in range(0, tl.cdiv(R, BLOCK_R)):
        offs_r = r_start * BLOCK_R + tl.arange(0, BLOCK_R)
        mask = offs_r < R
        acc = tl.load(acc_ptr + row * stride_am + offs_r * stride_ar, mask=mask, other=0).to(tl.float32)
        w_scale = tl.load(w_scale_ptr + offs_r, mask=mask, other=0.0).to(tl.float32)
        val = acc * x_scale * w_scale
        scaled = val / scale
        rounded = tl.where(scaled >= 0, scaled + 0.5, scaled - 0.5).to(tl.int32)
        clamped = tl.maximum(tl.minimum(rounded, MAXQ), -(MAXQ + 1))
        tl.store(q_ptr + row * stride_qm + offs_r * stride_qr, clamped.to(tl.int8), mask=mask)

def _w8a8_svd_matmul(x_q: torch.Tensor, w_q: torch.Tensor, kernel) -> torch.Tensor:
    x_q = x_q.contiguous()
    w_q = w_q.contiguous()
    m, k = x_q.shape
    n = w_q.shape[0]
    out = torch.empty((m, n), device=x_q.device, dtype=torch.int32)
    grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]), triton.cdiv(n, META["BLOCK_N"]))
    kernel[grid](
        x_q,
        w_q,
        out,
        m,
        n,
        k,
        x_q.stride(0),
        x_q.stride(1),
        w_q.stride(0),
        w_q.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out

def svd_rank_dequant_quantize(
    acc: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bits: int = 8,
    kernel=_svd_rank_dequant_quant_up_kernel,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m, r = acc.shape
    q = torch.empty((m, r), device=acc.device, dtype=torch.int8)
    scale = torch.empty((m,), device=acc.device, dtype=torch.float32)
    grid = (m,)
    kernel[grid](
        acc,
        q,
        scale,
        x_scale,
        w_scale,
        m,
        r,
        acc.stride(0),
        acc.stride(1),
        q.stride(0),
        q.stride(1),
        MAXQ=_maxq(bits),
    )
    return q, scale

def svd_out_from_ratio(out_dim: int, svd_ratio: float, round_to: int = 64) -> int:
    if svd_ratio <= 0.0:
        return 0
    if svd_ratio >= 1.0:
        return out_dim
    svd_out = int(out_dim * svd_ratio)
    if round_to > 1:
        svd_out = (svd_out // round_to) * round_to
    return max(0, min(out_dim, svd_out))

LOGGER = logging.getLogger(__name__)

_MIXED_GATEUP_V8_FFN_CACHE: Dict[Tuple[int, Tuple[int, ...], float, float, int], torch.nn.Module] = {}

@triton.autotune(
    configs=get_w8a8_dot_up_matmul_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def _w8a8_dual_matmul_svd_kernel(
    a_ptr,
    w0_ptr,
    w1_ptr,
    out0_ptr,
    out1_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_w0n,
    stride_w0k,
    stride_w1n,
    stride_w1k,
    stride_o0m,
    stride_o0n,
    stride_o1m,
    stride_o1n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0,
        ).to(tl.int8)
        w0 = tl.load(
            w0_ptr + offs_k[:, None] * stride_w0k + offs_n[None, :] * stride_w0n,
            mask=mask_k[:, None] & (offs_n[None, :] < N),
            other=0,
        ).to(tl.int8)
        w1 = tl.load(
            w1_ptr + offs_k[:, None] * stride_w1k + offs_n[None, :] * stride_w1n,
            mask=mask_k[:, None] & (offs_n[None, :] < N),
            other=0,
        ).to(tl.int8)
        acc0 += tl.dot(a, w0)
        acc1 += tl.dot(a, w1)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out0_ptr + offs_m[:, None] * stride_o0m + offs_n[None, :] * stride_o0n, acc0, mask=mask)
    tl.store(out1_ptr + offs_m[:, None] * stride_o1m + offs_n[None, :] * stride_o1n, acc1, mask=mask)

@triton.autotune(
    configs=get_w8a8_dot_up_matmul_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def _w8a8_dual_matmul_dense_kernel(
    a_ptr,
    w0_ptr,
    w1_ptr,
    out0_ptr,
    out1_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_w0n,
    stride_w0k,
    stride_w1n,
    stride_w1k,
    stride_o0m,
    stride_o0n,
    stride_o1m,
    stride_o1n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0,
        ).to(tl.int8)
        w0 = tl.load(
            w0_ptr + offs_k[:, None] * stride_w0k + offs_n[None, :] * stride_w0n,
            mask=mask_k[:, None] & (offs_n[None, :] < N),
            other=0,
        ).to(tl.int8)
        w1 = tl.load(
            w1_ptr + offs_k[:, None] * stride_w1k + offs_n[None, :] * stride_w1n,
            mask=mask_k[:, None] & (offs_n[None, :] < N),
            other=0,
        ).to(tl.int8)
        acc0 += tl.dot(a, w0)
        acc1 += tl.dot(a, w1)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out0_ptr + offs_m[:, None] * stride_o0m + offs_n[None, :] * stride_o0n, acc0, mask=mask)
    tl.store(out1_ptr + offs_m[:, None] * stride_o1m + offs_n[None, :] * stride_o1n, acc1, mask=mask)

def w8a8_dual_matmul(
    x_q: torch.Tensor,
    w0_q: torch.Tensor,
    w1_q: torch.Tensor,
    kernel=_w8a8_dual_matmul_svd_kernel,
):
    if x_q.ndim != 2 or w0_q.ndim != 2 or w1_q.ndim != 2:
        raise ValueError("w8a8_dual_matmul expects 2D tensors")
    if x_q.dtype != torch.int8 or w0_q.dtype != torch.int8 or w1_q.dtype != torch.int8:
        raise ValueError("w8a8_dual_matmul expects int8 tensors")
    m, k = x_q.shape
    n0, k0 = w0_q.shape
    n1, k1 = w1_q.shape
    if k != k0 or k != k1:
        raise ValueError("Weight/input K dimensions must match")
    if n0 != n1:
        raise ValueError("Dual matmul expects the two outputs to have the same width")

    out0 = torch.empty((m, n0), device=x_q.device, dtype=torch.int32)
    out1 = torch.empty((m, n0), device=x_q.device, dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n0, meta["BLOCK_N"]))
    kernel[grid](
        x_q,
        w0_q,
        w1_q,
        out0,
        out1,
        m,
        n0,
        k,
        x_q.stride(0),
        x_q.stride(1),
        w0_q.stride(0),
        w0_q.stride(1),
        w1_q.stride(0),
        w1_q.stride(1),
        out0.stride(0),
        out0.stride(1),
        out1.stride(0),
        out1.stride(1),
    )
    return out0, out1

@triton.autotune(
    configs=get_dual_quantize_vec_autotune_config(),
    key=["N", "MAXQ"],
)
@triton.jit
def _dual_quantize_with_scale_vec_kernel(
    x0_ptr,
    x1_ptr,
    q0_ptr,
    q1_ptr,
    scale0_ptr,
    scale1_ptr,
    M,
    N,
    stride_x0m,
    stride_x0n,
    stride_x1m,
    stride_x1n,
    stride_q0m,
    stride_q0n,
    stride_q1m,
    stride_q1n,
    MAXQ: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(axis=0)
    if row >= M:
        return
    scale0 = tl.load(scale0_ptr + row).to(tl.float32)
    scale1 = tl.load(scale1_ptr + row).to(tl.float32)
    for n_start in range(0, tl.cdiv(N, BLOCK_N)):
        offs_n = n_start * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        x0 = tl.load(x0_ptr + row * stride_x0m + offs_n * stride_x0n, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + row * stride_x1m + offs_n * stride_x1n, mask=mask, other=0.0).to(tl.float32)
        scaled0 = x0 / scale0
        scaled1 = x1 / scale1
        rounded0 = tl.where(scaled0 >= 0, scaled0 + 0.5, scaled0 - 0.5).to(tl.int32)
        rounded1 = tl.where(scaled1 >= 0, scaled1 + 0.5, scaled1 - 0.5).to(tl.int32)
        clamped0 = tl.maximum(tl.minimum(rounded0, MAXQ), -(MAXQ + 1))
        clamped1 = tl.maximum(tl.minimum(rounded1, MAXQ), -(MAXQ + 1))
        tl.store(q0_ptr + row * stride_q0m + offs_n * stride_q0n, clamped0.to(tl.int8), mask=mask)
        tl.store(q1_ptr + row * stride_q1m + offs_n * stride_q1n, clamped1.to(tl.int8), mask=mask)

@triton.autotune(
    configs=get_dual_quantize_vec_autotune_config(),
    key=["N", "MAXQ"],
)
@triton.jit
def _dual_quantize_with_shared_scale_vec_kernel(
    x0_ptr,
    x1_ptr,
    q0_ptr,
    q1_ptr,
    scale_ptr,
    M,
    N,
    stride_x0m,
    stride_x0n,
    stride_x1m,
    stride_x1n,
    stride_q0m,
    stride_q0n,
    stride_q1m,
    stride_q1n,
    MAXQ: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(axis=0)
    if row >= M:
        return
    scale = tl.load(scale_ptr + row).to(tl.float32)
    for n_start in range(0, tl.cdiv(N, BLOCK_N)):
        offs_n = n_start * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        x0 = tl.load(x0_ptr + row * stride_x0m + offs_n * stride_x0n, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + row * stride_x1m + offs_n * stride_x1n, mask=mask, other=0.0).to(tl.float32)
        scaled0 = x0 / scale
        scaled1 = x1 / scale
        rounded0 = tl.where(scaled0 >= 0, scaled0 + 0.5, scaled0 - 0.5).to(tl.int32)
        rounded1 = tl.where(scaled1 >= 0, scaled1 + 0.5, scaled1 - 0.5).to(tl.int32)
        clamped0 = tl.maximum(tl.minimum(rounded0, MAXQ), -(MAXQ + 1))
        clamped1 = tl.maximum(tl.minimum(rounded1, MAXQ), -(MAXQ + 1))
        tl.store(q0_ptr + row * stride_q0m + offs_n * stride_q0n, clamped0.to(tl.int8), mask=mask)
        tl.store(q1_ptr + row * stride_q1m + offs_n * stride_q1n, clamped1.to(tl.int8), mask=mask)

@triton.autotune(
    configs=get_dual_svd_up_a_dequant_autotune_config(),
    key=["M", "R", "K"],
)
@triton.jit
def _dual_svd_up_a_matmul_dequant_tiled_kernel(
    x_q_ptr,
    up_a_ptr,
    gate_a_ptr,
    up_fp_ptr,
    gate_fp_ptr,
    up_tilemax_ptr,
    gate_tilemax_ptr,
    x_scale_ptr,
    up_w_scale_ptr,
    gate_w_scale_ptr,
    M,
    R,
    K,
    TILES_R,
    stride_xm,
    stride_xk,
    stride_uar,
    stride_uak,
    stride_gar,
    stride_gak,
    stride_ufm,
    stride_ufr,
    stride_gfm,
    stride_gfr,
    stride_utmm,
    stride_utmt,
    stride_gtmm,
    stride_gtmt,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_r = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = offs_m < M
    rank_mask = offs_r < R
    up_acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.int32)
    gate_acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.int32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_q = tl.load(
            x_q_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=row_mask[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.int8)
        up_a = tl.load(
            up_a_ptr + offs_k[:, None] * stride_uak + offs_r[None, :] * stride_uar,
            mask=mask_k[:, None] & rank_mask[None, :],
            other=0,
        ).to(tl.int8)
        gate_a = tl.load(
            gate_a_ptr + offs_k[:, None] * stride_gak + offs_r[None, :] * stride_gar,
            mask=mask_k[:, None] & rank_mask[None, :],
            other=0,
        ).to(tl.int8)
        up_acc += tl.dot(x_q, up_a)
        gate_acc += tl.dot(x_q, gate_a)

    x_scale = tl.load(x_scale_ptr + offs_m, mask=row_mask, other=0.0).to(tl.float32)[:, None]
    up_w_scale = tl.load(up_w_scale_ptr + offs_r, mask=rank_mask, other=0.0).to(tl.float32)[None, :]
    gate_w_scale = tl.load(gate_w_scale_ptr + offs_r, mask=rank_mask, other=0.0).to(tl.float32)[None, :]
    up_val = up_acc.to(tl.float32) * x_scale * up_w_scale
    gate_val = gate_acc.to(tl.float32) * x_scale * gate_w_scale
    up_val = tl.where(row_mask[:, None] & rank_mask[None, :], up_val, 0.0)
    gate_val = tl.where(row_mask[:, None] & rank_mask[None, :], gate_val, 0.0)
    tl.store(up_fp_ptr + offs_m[:, None] * stride_ufm + offs_r[None, :] * stride_ufr, up_val.to(tl.float16), mask=row_mask[:, None] & rank_mask[None, :])
    tl.store(gate_fp_ptr + offs_m[:, None] * stride_gfm + offs_r[None, :] * stride_gfr, gate_val.to(tl.float16), mask=row_mask[:, None] & rank_mask[None, :])
    up_tile_max = tl.max(tl.abs(up_val), axis=1)
    gate_tile_max = tl.max(tl.abs(gate_val), axis=1)
    tl.store(up_tilemax_ptr + offs_m * stride_utmm + pid_r * stride_utmt, up_tile_max, mask=row_mask)
    tl.store(gate_tilemax_ptr + offs_m * stride_gtmm + pid_r * stride_gtmt, gate_tile_max, mask=row_mask)

@triton.autotune(
    configs=get_svd_down_b_dequant_autotune_config(),
    key=["M", "N", "R"],
)
@triton.jit
def _svd_down_b_matmul_dequant_kernel(
    rank_q_ptr,
    b_ptr,
    out_fp_ptr,
    rank_scale_ptr,
    b_scale_ptr,
    M,
    N,
    R,
    stride_rqm,
    stride_rqr,
    stride_bn,
    stride_br,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = offs_m < M
    col_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for r_start in range(0, tl.cdiv(R, BLOCK_R)):
        offs_r = r_start * BLOCK_R + tl.arange(0, BLOCK_R)
        rank_mask = offs_r < R
        rank_q = tl.load(
            rank_q_ptr + offs_m[:, None] * stride_rqm + offs_r[None, :] * stride_rqr,
            mask=row_mask[:, None] & rank_mask[None, :],
            other=0,
        ).to(tl.int8)
        b = tl.load(
            b_ptr + offs_r[:, None] * stride_br + offs_n[None, :] * stride_bn,
            mask=rank_mask[:, None] & col_mask[None, :],
            other=0,
        ).to(tl.int8)
        acc += tl.dot(rank_q, b)

    rank_scale = tl.load(rank_scale_ptr + offs_m, mask=row_mask, other=0.0).to(tl.float32)[:, None]
    b_scale = tl.load(b_scale_ptr + offs_n, mask=col_mask, other=0.0).to(tl.float32)[None, :]
    out = acc.to(tl.float32) * rank_scale * b_scale
    mask = row_mask[:, None] & col_mask[None, :]
    tl.store(out_fp_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, out.to(tl.float16), mask=mask)

@triton.autotune(
    configs=get_dense_down_dequant_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def _dense_down_matmul_dequant_kernel(
    x_q_ptr,
    w_ptr,
    out_fp_ptr,
    x_scale_ptr,
    w_scale_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = offs_m < M
    col_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_q = tl.load(
            x_q_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=row_mask[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.int8)
        w = tl.load(
            w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=mask_k[:, None] & col_mask[None, :],
            other=0,
        ).to(tl.int8)
        acc += tl.dot(x_q, w)

    x_scale = tl.load(x_scale_ptr + offs_m, mask=row_mask, other=0.0).to(tl.float32)[:, None]
    w_scale = tl.load(w_scale_ptr + offs_n, mask=col_mask, other=0.0).to(tl.float32)[None, :]
    out = acc.to(tl.float32) * x_scale * w_scale
    mask = row_mask[:, None] & col_mask[None, :]
    tl.store(out_fp_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, out.to(tl.float16), mask=mask)

@triton.autotune(
    configs=get_swiglu_quant_autotune_config(),
    key=["M", "N", "S", "MAXQ"],
)
@triton.jit
def _mixed_dequant_swiglu_quant_kernel(
    up_svd_acc_ptr,
    gate_svd_acc_ptr,
    up_dense_acc_ptr,
    gate_dense_acc_ptr,
    out_q_ptr,
    out_scale_ptr,
    up_svd_x_scale_ptr,
    gate_svd_x_scale_ptr,
    dense_x_scale_ptr,
    up_svd_w_scale_ptr,
    gate_svd_w_scale_ptr,
    up_dense_w_scale_ptr,
    gate_dense_w_scale_ptr,
    M,
    N,
    S,
    D,
    stride_uss_m,
    stride_uss_n,
    stride_gss_m,
    stride_gss_n,
    stride_uds_m,
    stride_uds_n,
    stride_gds_m,
    stride_gds_n,
    stride_om,
    stride_on,
    MAXQ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = offs_m < M

    row_max = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for n_start in range(0, tl.cdiv(N, BLOCK_N)):
        offs_n = n_start * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = offs_n < N
        mask = row_mask[:, None] & col_mask[None, :]

        svd_mask = offs_n < S
        dense_idx = offs_n - S
        dense_mask = (dense_idx >= 0) & (dense_idx < D)

        up_svd_acc = tl.load(
            up_svd_acc_ptr + offs_m[:, None] * stride_uss_m + offs_n[None, :] * stride_uss_n,
            mask=row_mask[:, None] & svd_mask[None, :],
            other=0,
        ).to(tl.float32)
        gate_svd_acc = tl.load(
            gate_svd_acc_ptr + offs_m[:, None] * stride_gss_m + offs_n[None, :] * stride_gss_n,
            mask=row_mask[:, None] & svd_mask[None, :],
            other=0,
        ).to(tl.float32)
        up_dense_acc = tl.load(
            up_dense_acc_ptr + offs_m[:, None] * stride_uds_m + dense_idx[None, :] * stride_uds_n,
            mask=row_mask[:, None] & dense_mask[None, :],
            other=0,
        ).to(tl.float32)
        gate_dense_acc = tl.load(
            gate_dense_acc_ptr + offs_m[:, None] * stride_gds_m + dense_idx[None, :] * stride_gds_n,
            mask=row_mask[:, None] & dense_mask[None, :],
            other=0,
        ).to(tl.float32)

        up_svd_x_scale = tl.load(up_svd_x_scale_ptr + offs_m, mask=row_mask, other=0.0).to(tl.float32)[:, None]
        gate_svd_x_scale = tl.load(gate_svd_x_scale_ptr + offs_m, mask=row_mask, other=0.0).to(tl.float32)[:, None]
        dense_x_scale = tl.load(dense_x_scale_ptr + offs_m, mask=row_mask, other=0.0).to(tl.float32)[:, None]

        up_svd_w_scale = tl.load(up_svd_w_scale_ptr + offs_n, mask=svd_mask, other=0.0).to(tl.float32)[None, :]
        gate_svd_w_scale = tl.load(gate_svd_w_scale_ptr + offs_n, mask=svd_mask, other=0.0).to(tl.float32)[None, :]
        up_dense_w_scale = tl.load(up_dense_w_scale_ptr + dense_idx, mask=dense_mask, other=0.0).to(tl.float32)[None, :]
        gate_dense_w_scale = tl.load(gate_dense_w_scale_ptr + dense_idx, mask=dense_mask, other=0.0).to(tl.float32)[None, :]

        up_svd = up_svd_acc * up_svd_x_scale * up_svd_w_scale
        gate_svd = gate_svd_acc * gate_svd_x_scale * gate_svd_w_scale
        up_dense = up_dense_acc * dense_x_scale * up_dense_w_scale
        gate_dense = gate_dense_acc * dense_x_scale * gate_dense_w_scale

        up = tl.where(svd_mask[None, :], up_svd, up_dense)
        gate = tl.where(svd_mask[None, :], gate_svd, gate_dense)
        silu_gate = gate * tl.sigmoid(gate)
        swiglu = up * silu_gate
        swiglu = tl.where(mask, swiglu, 0.0)
        row_max = tl.maximum(row_max, tl.max(tl.abs(swiglu), axis=1))

    out_scale = tl.maximum(row_max / MAXQ, 1e-8)
    tl.store(out_scale_ptr + offs_m, out_scale, mask=row_mask)

    for n_start in range(0, tl.cdiv(N, BLOCK_N)):
        offs_n = n_start * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = offs_n < N
        mask = row_mask[:, None] & col_mask[None, :]

        svd_mask = offs_n < S
        dense_idx = offs_n - S
        dense_mask = (dense_idx >= 0) & (dense_idx < D)

        up_svd_acc = tl.load(
            up_svd_acc_ptr + offs_m[:, None] * stride_uss_m + offs_n[None, :] * stride_uss_n,
            mask=row_mask[:, None] & svd_mask[None, :],
            other=0,
        ).to(tl.float32)
        gate_svd_acc = tl.load(
            gate_svd_acc_ptr + offs_m[:, None] * stride_gss_m + offs_n[None, :] * stride_gss_n,
            mask=row_mask[:, None] & svd_mask[None, :],
            other=0,
        ).to(tl.float32)
        up_dense_acc = tl.load(
            up_dense_acc_ptr + offs_m[:, None] * stride_uds_m + dense_idx[None, :] * stride_uds_n,
            mask=row_mask[:, None] & dense_mask[None, :],
            other=0,
        ).to(tl.float32)
        gate_dense_acc = tl.load(
            gate_dense_acc_ptr + offs_m[:, None] * stride_gds_m + dense_idx[None, :] * stride_gds_n,
            mask=row_mask[:, None] & dense_mask[None, :],
            other=0,
        ).to(tl.float32)

        up_svd_x_scale = tl.load(up_svd_x_scale_ptr + offs_m, mask=row_mask, other=0.0).to(tl.float32)[:, None]
        gate_svd_x_scale = tl.load(gate_svd_x_scale_ptr + offs_m, mask=row_mask, other=0.0).to(tl.float32)[:, None]
        dense_x_scale = tl.load(dense_x_scale_ptr + offs_m, mask=row_mask, other=0.0).to(tl.float32)[:, None]

        up_svd_w_scale = tl.load(up_svd_w_scale_ptr + offs_n, mask=svd_mask, other=0.0).to(tl.float32)[None, :]
        gate_svd_w_scale = tl.load(gate_svd_w_scale_ptr + offs_n, mask=svd_mask, other=0.0).to(tl.float32)[None, :]
        up_dense_w_scale = tl.load(up_dense_w_scale_ptr + dense_idx, mask=dense_mask, other=0.0).to(tl.float32)[None, :]
        gate_dense_w_scale = tl.load(gate_dense_w_scale_ptr + dense_idx, mask=dense_mask, other=0.0).to(tl.float32)[None, :]

        up_svd = up_svd_acc * up_svd_x_scale * up_svd_w_scale
        gate_svd = gate_svd_acc * gate_svd_x_scale * gate_svd_w_scale
        up_dense = up_dense_acc * dense_x_scale * up_dense_w_scale
        gate_dense = gate_dense_acc * dense_x_scale * gate_dense_w_scale

        up = tl.where(svd_mask[None, :], up_svd, up_dense)
        gate = tl.where(svd_mask[None, :], gate_svd, gate_dense)
        silu_gate = gate * tl.sigmoid(gate)
        swiglu = up * silu_gate
        scaled = swiglu / out_scale[:, None]
        rounded = tl.where(scaled >= 0, scaled + 0.5, scaled - 0.5).to(tl.int32)
        clamped = tl.maximum(tl.minimum(rounded, MAXQ), -(MAXQ + 1))
        tl.store(out_q_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, clamped.to(tl.int8), mask=mask)

class DenseQuantLinearW8A8(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("q", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.ones((out_features,), dtype=torch.float32))

    @classmethod
    def from_weight(cls, weight: torch.Tensor):
        out = cls(weight.shape[1], weight.shape[0]).to(weight.device)
        q, scale = quantize_weights_per_channel(weight, 8)
        out.q.copy_(q)
        out.scale.copy_(scale)
        return out

class SVDQuantLinearW8A8(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.register_buffer("a_q", torch.zeros((rank, in_features), dtype=torch.int8))
        self.register_buffer("b_q", torch.zeros((out_features, rank), dtype=torch.int8))
        self.register_buffer("a_scale", torch.ones((rank,), dtype=torch.float32))
        self.register_buffer("b_scale", torch.ones((out_features,), dtype=torch.float32))

    @classmethod
    def from_shape(cls, in_features: int, out_features: int, rank_ratio: float, round_to: int, device, dtype):
        rank = _rounded_rank(in_features, out_features, rank_ratio, round_to)
        out = cls(in_features, out_features, rank).to(device)
        dummy_weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
        a, b = _init_low_rank_factors(dummy_weight, rank)
        a_q, a_scale = quantize_weights_per_channel(a, 8)
        b_q, b_scale = quantize_weights_per_channel(b, 8)
        out.a_q.copy_(a_q)
        out.a_scale.copy_(a_scale)
        out.b_q.copy_(b_q)
        out.b_scale.copy_(b_scale)
        return out

class MixedSVDDenseLinearW8A8(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, svd_out: int, dense_out: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.svd_out = svd_out
        self.dense_out = dense_out
        self.svd_branch: SVDQuantLinearW8A8 | None = None
        self.dense_branch: DenseQuantLinearW8A8 | None = None

    @classmethod
    def from_weight(cls, weight: torch.Tensor, svd_ratio: float, rank_ratio: float, round_to: int):
        out_features, in_features = weight.shape
        svd_out = svd_out_from_ratio(out_features, svd_ratio, round_to=round_to)
        dense_out = out_features - svd_out
        out = cls(in_features, out_features, svd_out, dense_out).to(weight.device)
        if svd_out > 0:
            out.svd_branch = SVDQuantLinearW8A8.from_shape(
                in_features,
                svd_out,
                rank_ratio,
                round_to,
                device=weight.device,
                dtype=weight.dtype,
            )
        if dense_out > 0:
            out.dense_branch = DenseQuantLinearW8A8.from_weight(weight[svd_out:, :].contiguous())
        return out

    def output_scale(self) -> torch.Tensor:
        parts = []
        if self.svd_branch is not None:
            parts.append(self.svd_branch.b_scale)
        if self.dense_branch is not None:
            parts.append(self.dense_branch.scale)
        return torch.cat(parts, dim=0) if parts else torch.empty((0,), device=self.svd_branch.a_q.device if self.svd_branch else self.dense_branch.q.device)

def mixed_dequant_swiglu_quantize(
    up_state: Dict[str, torch.Tensor],
    gate_state: Dict[str, torch.Tensor],
    up_proj: MixedSVDDenseLinearW8A8,
    gate_proj: MixedSVDDenseLinearW8A8,
    x_scale: torch.Tensor,
    bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if up_proj.out_features != gate_proj.out_features:
        raise ValueError("up/gate mixed projections must have the same output size")
    m = x_scale.shape[0]
    n = up_proj.out_features
    s = up_proj.svd_out
    d = up_proj.dense_out
    if s != gate_proj.svd_out or d != gate_proj.dense_out:
        raise ValueError("up/gate mixed branch layouts must match")

    q = torch.empty((m, n), device=x_scale.device, dtype=torch.int8)
    scale = torch.empty((m,), device=x_scale.device, dtype=torch.float32)

    up_svd_acc = up_state.get("svd_out_acc")
    gate_svd_acc = gate_state.get("svd_out_acc")
    up_dense_acc = up_state.get("dense_out_acc")
    gate_dense_acc = gate_state.get("dense_out_acc")

    dummy_i32 = torch.empty((m, 1), device=x_scale.device, dtype=torch.int32)
    dummy_f32 = torch.empty((1,), device=x_scale.device, dtype=torch.float32)
    up_svd_acc = up_svd_acc if up_svd_acc is not None else dummy_i32
    gate_svd_acc = gate_svd_acc if gate_svd_acc is not None else dummy_i32
    up_dense_acc = up_dense_acc if up_dense_acc is not None else dummy_i32
    gate_dense_acc = gate_dense_acc if gate_dense_acc is not None else dummy_i32

    up_svd_x_scale = up_state.get("svd_rank_scale", x_scale)
    gate_svd_x_scale = gate_state.get("svd_rank_scale", x_scale)
    up_svd_w_scale = up_proj.svd_branch.b_scale if up_proj.svd_branch is not None else dummy_f32
    gate_svd_w_scale = gate_proj.svd_branch.b_scale if gate_proj.svd_branch is not None else dummy_f32
    up_dense_w_scale = up_proj.dense_branch.scale if up_proj.dense_branch is not None else dummy_f32
    gate_dense_w_scale = gate_proj.dense_branch.scale if gate_proj.dense_branch is not None else dummy_f32

    grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]),)
    _mixed_dequant_swiglu_quant_kernel[grid](
        up_svd_acc,
        gate_svd_acc,
        up_dense_acc,
        gate_dense_acc,
        q,
        scale,
        up_svd_x_scale,
        gate_svd_x_scale,
        x_scale,
        up_svd_w_scale,
        gate_svd_w_scale,
        up_dense_w_scale,
        gate_dense_w_scale,
        m,
        n,
        s,
        d,
        up_svd_acc.stride(0),
        up_svd_acc.stride(1),
        gate_svd_acc.stride(0),
        gate_svd_acc.stride(1),
        up_dense_acc.stride(0),
        up_dense_acc.stride(1),
        gate_dense_acc.stride(0),
        gate_dense_acc.stride(1),
        q.stride(0),
        q.stride(1),
        MAXQ=127 if bits == 8 else 7,
    )
    return q, scale

def dual_svd_up_a_dequant_compute_max_tiled(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    up_proj: MixedSVDDenseLinearW8A8,
    gate_proj: MixedSVDDenseLinearW8A8,
    bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if up_proj.svd_branch is None or gate_proj.svd_branch is None:
        raise ValueError("dual svd up-a tiled path requires both svd branches")
    if up_proj.svd_branch.rank != gate_proj.svd_branch.rank:
        raise ValueError("up/gate svd ranks must match")
    m, k = x_q.shape
    r = up_proj.svd_branch.rank
    # Scratch row-max storage must cover the smallest autotuned BLOCK_R.
    tiles_r = triton.cdiv(r, 32)
    up_fp = torch.empty((m, r), device=x_q.device, dtype=torch.float16)
    gate_fp = torch.empty((m, r), device=x_q.device, dtype=torch.float16)
    up_tilemax = torch.empty((m, tiles_r), device=x_q.device, dtype=torch.float32)
    gate_tilemax = torch.empty((m, tiles_r), device=x_q.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(r, meta["BLOCK_R"]))
    _dual_svd_up_a_matmul_dequant_tiled_kernel[grid](
        x_q,
        up_proj.svd_branch.a_q,
        gate_proj.svd_branch.a_q,
        up_fp,
        gate_fp,
        up_tilemax,
        gate_tilemax,
        x_scale,
        up_proj.svd_branch.a_scale,
        gate_proj.svd_branch.a_scale,
        m,
        r,
        k,
        tiles_r,
        x_q.stride(0),
        x_q.stride(1),
        up_proj.svd_branch.a_q.stride(0),
        up_proj.svd_branch.a_q.stride(1),
        gate_proj.svd_branch.a_q.stride(0),
        gate_proj.svd_branch.a_q.stride(1),
        up_fp.stride(0),
        up_fp.stride(1),
        gate_fp.stride(0),
        gate_fp.stride(1),
        up_tilemax.stride(0),
        up_tilemax.stride(1),
        gate_tilemax.stride(0),
        gate_tilemax.stride(1),
    )
    maxq = 127.0 if bits == 8 else 7.0
    up_scale = torch.clamp(up_tilemax.amax(dim=1) / maxq, min=1e-8)
    gate_scale = torch.clamp(gate_tilemax.amax(dim=1) / maxq, min=1e-8)
    return up_fp, up_scale, gate_fp, gate_scale

def svd_down_b_matmul_dequant(
    rank_q: torch.Tensor,
    rank_scale: torch.Tensor,
    proj: MixedSVDDenseLinearW8A8,
) -> torch.Tensor:
    if proj.svd_branch is None:
        raise ValueError("svd down-b fused path requires an svd branch")
    m, r = rank_q.shape
    n = proj.svd_out
    out_fp = torch.empty((m, n), device=rank_q.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))
    _svd_down_b_matmul_dequant_kernel[grid](
        rank_q,
        proj.svd_branch.b_q,
        out_fp,
        rank_scale,
        proj.svd_branch.b_scale,
        m,
        n,
        r,
        rank_q.stride(0),
        rank_q.stride(1),
        proj.svd_branch.b_q.stride(0),
        proj.svd_branch.b_q.stride(1),
        out_fp.stride(0),
        out_fp.stride(1),
    )
    return out_fp

def dense_down_matmul_dequant(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    proj: MixedSVDDenseLinearW8A8,
) -> torch.Tensor:
    if proj.dense_branch is None:
        raise ValueError("dense down fused path requires a dense branch")
    m, k = x_q.shape
    n = proj.dense_out
    out_fp = torch.empty((m, n), device=x_q.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))
    _dense_down_matmul_dequant_kernel[grid](
        x_q,
        proj.dense_branch.q,
        out_fp,
        x_scale,
        proj.dense_branch.scale,
        m,
        n,
        k,
        x_q.stride(0),
        x_q.stride(1),
        proj.dense_branch.q.stride(0),
        proj.dense_branch.q.stride(1),
        out_fp.stride(0),
        out_fp.stride(1),
    )
    return out_fp

def dual_quantize_mid_with_scale(
    x0_fp: torch.Tensor,
    x1_fp: torch.Tensor,
    scale0: torch.Tensor,
    scale1: torch.Tensor,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x0_fp.shape != x1_fp.shape:
        raise ValueError("dual quantize expects matching tensor shapes")
    m, n = x0_fp.shape
    q0 = torch.empty((m, n), device=x0_fp.device, dtype=torch.int8)
    q1 = torch.empty((m, n), device=x1_fp.device, dtype=torch.int8)
    grid = (m,)
    _dual_quantize_with_scale_vec_kernel[grid](
        x0_fp,
        x1_fp,
        q0,
        q1,
        scale0,
        scale1,
        m,
        n,
        x0_fp.stride(0),
        x0_fp.stride(1),
        x1_fp.stride(0),
        x1_fp.stride(1),
        q0.stride(0),
        q0.stride(1),
        q1.stride(0),
        q1.stride(1),
        MAXQ=127 if bits == 8 else 7,
    )
    return q0, q1

def dual_quantize_mid_with_shared_scale(
    x0_fp: torch.Tensor,
    x1_fp: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x0_fp.shape != x1_fp.shape:
        raise ValueError("dual quantize expects matching tensor shapes")
    m, n = x0_fp.shape
    q0 = torch.empty((m, n), device=x0_fp.device, dtype=torch.int8)
    q1 = torch.empty((m, n), device=x1_fp.device, dtype=torch.int8)
    grid = (m,)
    _dual_quantize_with_shared_scale_vec_kernel[grid](
        x0_fp,
        x1_fp,
        q0,
        q1,
        scale,
        m,
        n,
        x0_fp.stride(0),
        x0_fp.stride(1),
        x1_fp.stride(0),
        x1_fp.stride(1),
        q0.stride(0),
        q0.stride(1),
        q1.stride(0),
        q1.stride(1),
        MAXQ=127 if bits == 8 else 7,
    )
    return q0, q1

@triton.autotune(
    configs=get_mixed_gateup_swiglu_from_latents_autotune_config(),
    key=["M", "S", "R"],
)
@triton.jit
def _mixed_gateup_swiglu_from_latents_tiled_kernel(
    up_rank_q_ptr,
    gate_rank_q_ptr,
    up_svd_b_ptr,
    gate_svd_b_ptr,
    out_fp_ptr,
    tilemax_ptr,
    up_rank_scale_ptr,
    gate_rank_scale_ptr,
    up_svd_b_scale_ptr,
    gate_svd_b_scale_ptr,
    M,
    S,
    R,
    TILES_S,
    stride_urm,
    stride_urr,
    stride_grm,
    stride_grr,
    stride_usb_n,
    stride_usb_r,
    stride_gsb_n,
    stride_gsb_r,
    stride_ofm,
    stride_ofn,
    stride_tmm,
    stride_tmt,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = offs_m < M
    col_mask = offs_n < S
    up_svd_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    gate_svd_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for r_start in range(0, tl.cdiv(R, BLOCK_R)):
        offs_r = r_start * BLOCK_R + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R
        up_r = tl.load(
            up_rank_q_ptr + offs_m[:, None] * stride_urm + offs_r[None, :] * stride_urr,
            mask=row_mask[:, None] & mask_r[None, :],
            other=0,
        ).to(tl.int8)
        gate_r = tl.load(
            gate_rank_q_ptr + offs_m[:, None] * stride_grm + offs_r[None, :] * stride_grr,
            mask=row_mask[:, None] & mask_r[None, :],
            other=0,
        ).to(tl.int8)
        up_b = tl.load(
            up_svd_b_ptr + offs_r[:, None] * stride_usb_r + offs_n[None, :] * stride_usb_n,
            mask=mask_r[:, None] & col_mask[None, :],
            other=0,
        ).to(tl.int8)
        gate_b = tl.load(
            gate_svd_b_ptr + offs_r[:, None] * stride_gsb_r + offs_n[None, :] * stride_gsb_n,
            mask=mask_r[:, None] & col_mask[None, :],
            other=0,
        ).to(tl.int8)
        up_svd_acc += tl.dot(up_r, up_b)
        gate_svd_acc += tl.dot(gate_r, gate_b)

    up_rank_scale = tl.load(up_rank_scale_ptr + offs_m, mask=row_mask, other=0.0).to(tl.float32)[:, None]
    gate_rank_scale = tl.load(gate_rank_scale_ptr + offs_m, mask=row_mask, other=0.0).to(tl.float32)[:, None]
    up_svd_w_scale = tl.load(up_svd_b_scale_ptr + offs_n, mask=col_mask, other=0.0).to(tl.float32)[None, :]
    gate_svd_w_scale = tl.load(gate_svd_b_scale_ptr + offs_n, mask=col_mask, other=0.0).to(tl.float32)[None, :]
    up = up_svd_acc.to(tl.float32) * up_rank_scale * up_svd_w_scale
    gate = gate_svd_acc.to(tl.float32) * gate_rank_scale * gate_svd_w_scale
    swiglu = up * (gate * tl.sigmoid(gate))
    swiglu = tl.where(row_mask[:, None] & col_mask[None, :], swiglu, 0.0)
    tl.store(out_fp_ptr + offs_m[:, None] * stride_ofm + offs_n[None, :] * stride_ofn, swiglu.to(tl.float16), mask=row_mask[:, None] & col_mask[None, :])
    tile_max = tl.max(tl.abs(swiglu), axis=1)
    tl.store(tilemax_ptr + offs_m * stride_tmm + pid_n * stride_tmt, tile_max, mask=row_mask)

def gateup_svd_swiglu_from_latents_tiled(
    x_scale: torch.Tensor,
    up_aux: Dict[str, torch.Tensor],
    gate_aux: Dict[str, torch.Tensor],
    up_proj: MixedSVDDenseLinearW8A8,
    gate_proj: MixedSVDDenseLinearW8A8,
    bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if up_proj.out_features != gate_proj.out_features:
        raise ValueError("up/gate mixed projections must have the same output size")
    if up_proj.svd_out != gate_proj.svd_out:
        raise ValueError("up/gate svd branch layouts must match")
    if up_proj.svd_branch is None or gate_proj.svd_branch is None:
        raise ValueError("fused gate/up SVD branch requires both up/gate SVD branches")
    if "svd_rank_q" not in up_aux or "svd_rank_scale" not in up_aux:
        raise ValueError("up_aux must contain svd_rank_q and svd_rank_scale")
    if "svd_rank_q" not in gate_aux or "svd_rank_scale" not in gate_aux:
        raise ValueError("gate_aux must contain svd_rank_q and svd_rank_scale")

    m = x_scale.shape[0]
    s = up_proj.svd_out
    r = up_proj.svd_branch.rank
    # Scratch row-max storage must cover the smallest autotuned BLOCK_N.
    tiles_s = triton.cdiv(s, 64)

    up_rank_q = up_aux["svd_rank_q"]
    gate_rank_q = gate_aux["svd_rank_q"]
    up_rank_scale = up_aux["svd_rank_scale"]
    gate_rank_scale = gate_aux["svd_rank_scale"]
    up_svd_b = up_proj.svd_branch.b_q
    gate_svd_b = gate_proj.svd_branch.b_q
    up_svd_b_scale = up_proj.svd_branch.b_scale
    gate_svd_b_scale = gate_proj.svd_branch.b_scale
    up_svd_fp = torch.empty((m, s), device=x_scale.device, dtype=torch.float16)
    tilemax = torch.empty((m, tiles_s), device=x_scale.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(s, meta["BLOCK_N"]))
    _mixed_gateup_swiglu_from_latents_tiled_kernel[grid](
        up_rank_q,
        gate_rank_q,
        up_svd_b,
        gate_svd_b,
        up_svd_fp,
        tilemax,
        up_rank_scale,
        gate_rank_scale,
        up_svd_b_scale,
        gate_svd_b_scale,
        m,
        s,
        r,
        tiles_s,
        up_rank_q.stride(0),
        up_rank_q.stride(1),
        gate_rank_q.stride(0),
        gate_rank_q.stride(1),
        up_svd_b.stride(0),
        up_svd_b.stride(1),
        gate_svd_b.stride(0),
        gate_svd_b.stride(1),
        up_svd_fp.stride(0),
        up_svd_fp.stride(1),
        tilemax.stride(0),
        tilemax.stride(1),
    )
    maxq = 127.0 if bits == 8 else 7.0
    up_svd_scale = torch.clamp(tilemax.amax(dim=1) / maxq, min=1e-8)
    return up_svd_fp, up_svd_scale

def mixed_project_fp(state: Dict[str, torch.Tensor], proj: MixedSVDDenseLinearW8A8, x_scale: torch.Tensor) -> torch.Tensor:
    parts = []
    if proj.svd_branch is not None and "svd_out_acc" in state and "svd_rank_scale" in state:
        parts.append(dequantize_only(state["svd_out_acc"], state["svd_rank_scale"], proj.svd_branch.b_scale))
    if proj.dense_branch is not None and "dense_out_acc" in state:
        parts.append(dequantize_only(state["dense_out_acc"], x_scale, proj.dense_branch.scale))
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

class MixedFFNGateUpDenseLatentFused(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, svd_ratio: float, rank_ratio: float, round_to: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.svd_ratio = svd_ratio
        self.rank_ratio = rank_ratio
        self.round_to = round_to
        self.bits = 8
        self.eps = eps
        self.register_buffer("norm_weight", torch.ones(hidden_size, dtype=torch.float16))
        self.up_proj: MixedSVDDenseLinearW8A8
        self.gate_proj: MixedSVDDenseLinearW8A8
        self.down_proj: MixedSVDDenseLinearW8A8

    @classmethod
    def from_float(cls, module: SwiGLUFFN, svd_ratio: float, rank_ratio: float, round_to: int):
        out = cls(module.hidden_size, module.intermediate_size, svd_ratio, rank_ratio, round_to, eps=module.norm.variance_epsilon).to(module.up_proj.weight.device)
        out.norm_weight.copy_(module.norm.weight.detach())
        out.up_proj = MixedSVDDenseLinearW8A8.from_weight(module.up_proj.weight.detach(), svd_ratio, rank_ratio, round_to)
        out.gate_proj = MixedSVDDenseLinearW8A8.from_weight(module.gate_proj.weight.detach(), svd_ratio, rank_ratio, round_to)
        out.down_proj = MixedSVDDenseLinearW8A8.from_weight(module.down_proj.weight.detach(), svd_ratio, rank_ratio, round_to)
        return out

    def _fused_project_gateup_states(self, x_q: torch.Tensor, x_scale: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        up_aux: Dict[str, torch.Tensor] = {}
        gate_aux: Dict[str, torch.Tensor] = {}

        if self.up_proj.svd_branch is not None and self.gate_proj.svd_branch is not None:
            up_rank_fp, up_rank_scale, gate_rank_fp, gate_rank_scale = dual_svd_up_a_dequant_compute_max_tiled(
                x_q,
                x_scale,
                self.up_proj,
                self.gate_proj,
                self.bits,
            )
            up_rank_q, gate_rank_q = dual_quantize_mid_with_scale(
                up_rank_fp,
                gate_rank_fp,
                up_rank_scale,
                gate_rank_scale,
                self.bits,
            )
            up_aux.update(
                {
                    "svd_rank_fp": up_rank_fp,
                    "svd_rank_q": up_rank_q,
                    "svd_rank_scale": up_rank_scale,
                }
            )
            gate_aux.update(
                {
                    "svd_rank_fp": gate_rank_fp,
                    "svd_rank_q": gate_rank_q,
                    "svd_rank_scale": gate_rank_scale,
                }
            )
            svd_fp, svd_scale = gateup_svd_swiglu_from_latents_tiled(
                x_scale,
                up_aux,
                gate_aux,
                self.up_proj,
                self.gate_proj,
                self.bits,
            )
            up_aux.update({"svd_mid_fp": svd_fp, "svd_mid_scale": svd_scale})

        if self.up_proj.dense_branch is not None and self.gate_proj.dense_branch is not None:
            up_dense_acc, gate_dense_acc = w8a8_dual_matmul(
                x_q,
                self.up_proj.dense_branch.q,
                self.gate_proj.dense_branch.q,
                kernel=_w8a8_dual_matmul_dense_kernel,
            )
            up_aux["dense_out_acc"] = up_dense_acc
            gate_aux["dense_out_acc"] = gate_dense_acc
            dense_fp, dense_scale = swiglu_a2_compute_max(
                up_dense_acc,
                gate_dense_acc,
                x_scale,
                self.up_proj.dense_branch.scale,
                self.gate_proj.dense_branch.scale,
                self.bits,
            )
            up_aux.update({"dense_mid_fp": dense_fp, "dense_mid_scale": dense_scale})

        return up_aux, gate_aux

    def forward_stages(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        q_x, x_scale = rmsnorm_quantize_per_token(x, self.norm_weight, self.eps, self.bits)
        up_aux, gate_aux = self._fused_project_gateup_states(q_x, x_scale)

        svd_fp = up_aux["svd_mid_fp"]
        svd_scale = up_aux["svd_mid_scale"]
        dense_fp = up_aux["dense_mid_fp"]
        dense_scale = up_aux["dense_mid_scale"]

        mid_scale = torch.maximum(svd_scale, dense_scale)
        svd_q, dense_q = dual_quantize_mid_with_shared_scale(
            svd_fp,
            dense_fp,
            mid_scale,
            self.bits,
        )
        mid_q = torch.cat([svd_q, dense_q], dim=1).contiguous()

        mid_fp = None
        down_fp, down_aux = self._mixed_project_down_fp(mid_q, mid_scale, self.down_proj)
        out = down_fp + x.reshape(-1, x.shape[-1]) # notation: tensor bsz, seq dimension is flatten and then unflattened here
        return {
            "x_q": q_x,
            "x_scale": x_scale,
            "mid_fp": mid_fp,
            "mid_q": mid_q,
            "mid_scale": mid_scale,
            "down_fp": down_fp,
            "out": out.reshape_as(x),
            "svd_mid_fp": svd_fp,
            "svd_mid_scale": svd_scale,
            "dense_mid_fp": dense_fp,
            "dense_mid_scale": dense_scale,
            **{f"up_{k}": v for k, v in up_aux.items()},
            **{f"gate_{k}": v for k, v in gate_aux.items()},
            **{f"down_{k}": v for k, v in down_aux.items()},
        }

    def _mixed_project_down_fp(self, x_q: torch.Tensor, x_scale: torch.Tensor, proj: MixedSVDDenseLinearW8A8) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        parts = []
        aux: Dict[str, torch.Tensor] = {}
        if proj.svd_branch is not None:
            rank_acc = _w8a8_svd_matmul(x_q, proj.svd_branch.a_q, _w8a8_svd_down_a_matmul_kernel)
            rank_q, rank_scale = svd_rank_dequant_quantize(rank_acc, x_scale, proj.svd_branch.a_scale, self.bits, kernel=_svd_rank_dequant_quant_down_kernel)
            out_fp = svd_down_b_matmul_dequant(rank_q, rank_scale, proj)
            parts.append(out_fp)
            aux["svd_rank_acc"] = rank_acc
            aux["svd_rank_q"] = rank_q
            aux["svd_rank_scale"] = rank_scale
            aux["svd_out_fp"] = out_fp
        if proj.dense_branch is not None:
            out_fp = dense_down_matmul_dequant(x_q, x_scale, proj)
            parts.append(out_fp)
            aux["dense_out_fp"] = out_fp
        return (torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]), aux

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_stages(x)["out"]

def _mixed_module_cache_key(module: SwiGLUFFN, svd_ratio: float, rank_ratio: float, round_to: int):
    ptrs = (
        module.norm.weight.data_ptr(),
        module.up_proj.weight.data_ptr(),
        module.gate_proj.weight.data_ptr(),
        module.down_proj.weight.data_ptr(),
    )
    return id(module), ptrs, float(svd_ratio), float(rank_ratio), int(round_to)

def get_mixed_gateup_v8_ffn(module: SwiGLUFFN, svd_ratio: float | None = None, rank_ratio: float | None = None, round_to: int | None = None):
    svd_ratio = float(svd_ratio if svd_ratio is not None else getattr(module, "svd_ratio", 0.5))
    rank_ratio = float(rank_ratio if rank_ratio is not None else getattr(module, "rank_ratio", DEFAULT_RANK_RATIO))
    round_to = int(round_to if round_to is not None else getattr(module, "round_to", DEFAULT_ROUND_TO))
    key = _mixed_module_cache_key(module, svd_ratio, rank_ratio, round_to)
    cached = _MIXED_GATEUP_V8_FFN_CACHE.get(key)
    if cached is None:
        cached = MixedFFNGateUpDenseLatentFused.from_float(module, svd_ratio, rank_ratio, round_to)
        _MIXED_GATEUP_V8_FFN_CACHE[key] = cached
    return cached

def make_ffn_inputs(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    svd_ratio: float = 0.5,
    rank_ratio: float = DEFAULT_RANK_RATIO,
    round_to: int = DEFAULT_ROUND_TO,
):
    x, module = make_base_ffn_inputs(
        batch_size,
        seq_len,
        hidden_size,
        intermediate_size,
        seed=seed,
        dtype=dtype,
        device=device,
    )
    module.svd_ratio = float(svd_ratio)
    module.rank_ratio = float(rank_ratio)
    module.round_to = int(round_to)
    return x, module

def make_stage_callables(x: torch.Tensor, module, provider: str) -> Dict[str, Callable[[], object]]:
    if provider != "mixffn":
        raise ValueError(f"Stage callables only support mixffn, got {provider}")
    hybrid = get_mixed_gateup_v8_ffn(module)
    stages = hybrid.forward_stages(x)

    def gateup_svd():
        up_rank_fp, up_rank_scale, gate_rank_fp, gate_rank_scale = dual_svd_up_a_dequant_compute_max_tiled(
            stages["x_q"], stages["x_scale"], hybrid.up_proj, hybrid.gate_proj, hybrid.bits
        )
        up_rank_q, gate_rank_q = dual_quantize_mid_with_scale(up_rank_fp, gate_rank_fp, up_rank_scale, gate_rank_scale, hybrid.bits)
        return up_rank_q, up_rank_scale, gate_rank_q, gate_rank_scale

    def gateup_dense():
        return w8a8_dual_matmul(stages["x_q"], hybrid.up_proj.dense_branch.q, hybrid.gate_proj.dense_branch.q, kernel=_w8a8_dual_matmul_dense_kernel)

    def swiglu_quant():
        svd_fp, svd_scale = gateup_svd_swiglu_from_latents_tiled(
            stages["x_scale"],
            {k[3:]: v for k, v in stages.items() if k.startswith("up_")},
            {k[5:]: v for k, v in stages.items() if k.startswith("gate_")},
            hybrid.up_proj, hybrid.gate_proj, hybrid.bits,
        )
        dense_fp, dense_scale = swiglu_a2_compute_max(
            stages["up_dense_out_acc"], stages["gate_dense_out_acc"], stages["x_scale"],
            hybrid.up_proj.dense_branch.scale, hybrid.gate_proj.dense_branch.scale, hybrid.bits,
        )
        mid_scale = torch.maximum(svd_scale, dense_scale)
        svd_q, dense_q = dual_quantize_mid_with_shared_scale(svd_fp, dense_fp, mid_scale, hybrid.bits)
        return torch.cat([svd_q, dense_q], dim=1).contiguous(), mid_scale

    return {
        "rmsnorm_quant": lambda: rmsnorm_quantize_per_token(x, hybrid.norm_weight, hybrid.eps, hybrid.bits),
        "gateup_svd": gateup_svd,
        "gateup_dense": gateup_dense,
        "swiglu_quant": swiglu_quant,
        "down_proj": lambda: hybrid._mixed_project_down_fp(stages["mid_q"], stages["mid_scale"], hybrid.down_proj)[0],
        "residual_add": lambda: stages["down_fp"] + x.reshape(-1, x.shape[-1]),
    }

def benchmark_stage_breakdown(x: torch.Tensor, module, provider: str, repeat_time: int = 100, stage_mode: str = "cached") -> Dict[str, float]:
    if provider != "mixffn":
        raise ValueError(f"Stage timing only supports mixffn, got {provider}")
    if stage_mode not in {"cached", "cuda_graph"}:
        raise ValueError(f"Unsupported stage_mode: {stage_mode}")
    bench_mode = "cuda_graph" if stage_mode == "cuda_graph" else "eager"
    out = {name: bench_stage_callable(fn, repeat_time=repeat_time, bench_mode=bench_mode) for name, fn in make_stage_callables(x, module, provider).items()}
    out["sum_ms"] = float(sum(out.values()))
    ffn_stage_names = ("gateup_svd", "gateup_dense", "swiglu_quant", "down_proj")
    out["ffn_stage_sum_ms"] = float(sum(out[name] for name in ffn_stage_names))
    return out

def mixffn_ffn(x: torch.Tensor, module: SwiGLUFFN) -> torch.Tensor:
    return get_mixed_gateup_v8_ffn(module)(x)

_MIXFFN_KERNELS = (
    _rmsnorm_quant_kernel,
    _dual_svd_up_a_matmul_dequant_tiled_kernel,
    _dual_quantize_with_scale_vec_kernel,
    _dual_quantize_with_shared_scale_vec_kernel,
    _mixed_gateup_swiglu_from_latents_tiled_kernel,
    _swiglu_a2_max_vec_kernel,
    _w8a8_dual_matmul_dense_kernel,
    _w8a8_svd_down_a_matmul_kernel,
    _svd_rank_dequant_quant_down_kernel,
    _svd_down_b_matmul_dequant_kernel,
    _dense_down_matmul_dequant_kernel,
)

NAME = {"mixffn": mixffn_ffn}

KERNELS = {"mixffn": _MIXFFN_KERNELS}

DEFAULT_PROVIDERS = ["mixffn"]

FFN_STAGE_BENCH_PROVIDERS = {"mixffn"}
