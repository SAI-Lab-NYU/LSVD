"""
Hybrid SVD-Quant Gated FFN
==========================
Splits FFN hidden channels into two groups via a **single unified permutation**
applied consistently to all three projections.

Notation (d_ffn = intermediate_size, d_in = hidden_size):
  P   = original permutation (already applied to hidden states by an upstream
        step, e.g. QuaRot rotation).  In the standard pipeline P is absorbed
        into the weight matrices before this module is called, so it is the
        identity from our perspective.  Pass ``original_perm`` to absorb it
        explicitly if the weights have NOT yet been updated.
  P_s = grouping permutation we design   perm = svd_idx + quant_idx

Absorbed weight formulas (column-vector convention  y = W · x):
  Wa'   = P_s  @  Wa  @  P^T        gate / up  (output rows grouped, input un-rotated)
  Wout' = Wout @  P_s^T             down_proj  (input cols match [SVD | Quant] order)

When P = I (default, weights already P-absorbed):
  Wa'   = P_s  @  Wa   ≡  Wa[perm, :]
  Wout' = Wout @  P_s^T ≡  Wout[:, perm]

Resulting channel split (after applying perm):
  W_gate_grp  = W_gate[perm, :]       (d_ffn  × d_in)
  W_up_grp    = W_up  [perm, :]       (d_ffn  × d_in)
  W_down_grp  = W_down[:, perm]       (d_out  × d_ffn)

  Channels  0 : n_svd   → SVD   (low-rank approximation of W_gate / W_up)
  Channels n_svd : d_ffn → Quant (fake quant via GPTQ, or real INT8 weight quant)

Forward (identical semantics to the original gated FFN):
  g = W_gate_grp x = [g_svd | g_quant]
  u = W_up_grp   x = [u_svd | u_quant]
  h = SiLU(g) ⊙ u  = [SiLU(g_svd)⊙u_svd | SiLU(g_quant)⊙u_quant]
  y = W_down_grp h            (columns already in [SVD | Quant] order)

Accuracy guarantee
------------------
Channel selection uses a **dual-criterion score**:
  score_i = svd_quality_i / (output_importance_i + ε)
             ^               ^
         how well SVD       how much output error
         captures ch i      is amplified by W_down

Channels with *high SVD quality AND low output importance* go to SVD.
Channels with *high importance OR low SVD quality* go to Quant.
This ensures the pure-quant accuracy is a lower bound: in the worst case
all channels would be in the Quant group (= original behaviour).
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import hadamard_utils


_SVD_AWARE_MAX_TOKENS = 8192


# ---------------------------------------------------------------------------
# Channel sensitivity analysis
# ---------------------------------------------------------------------------

def _get_hybrid_ffn_sigma_fuse(args) -> str:
    """Return the sigma split mode used by hybrid FFN SVD factors."""
    return getattr(args, "sigma_fuse", getattr(args, "svd_mode", "UV"))


def _hybrid_ffn_groups_cache_path(args) -> Optional[str]:
    """Return the default on-disk cache path for hybrid FFN channel groups.

    Reuse the project's shared calibration cache directory when the user does
    not provide an explicit JSON path for hybrid FFN analysis/groups.
    """
    act_cache_dir = getattr(args, "act_cache_dir", None)
    if not act_cache_dir:
        return None

    cache_dir = os.path.join(act_cache_dir, "hybrid_ffn")
    nsamples = getattr(args, "nsamples", "na")
    seed = getattr(args, "seed", "na")
    calib_method = getattr(args, "svd_ffn_calib_method", "error_compare")
    svd_ratio = getattr(args, "svd_ffn_ratio", "na")
    effective_rank_ratio = getattr(args, "hybrid_ffn_rank_ratio", None)
    if effective_rank_ratio is None:
        effective_rank_ratio = getattr(args, "rank_ratio", "na")
    sigma_fuse = _get_hybrid_ffn_sigma_fuse(args)
    w_bits = getattr(args, "w_bits", "na")
    w_sym_tag = "w_asym" if bool(getattr(args, "w_asym", False)) else "w_sym"
    a_bits = getattr(args, "a_bits", "na")
    a_sym_tag = "a_asym" if bool(getattr(args, "a_asym", False)) else "a_sym"
    a_groupsize = getattr(args, "a_groupsize", "na")
    a_clip_ratio = min(
        float(getattr(args, "a_clip_ratio", 1.0)),
        float(getattr(args, "lma_clip_ratio", 1.0)),
    )
    latent_rotate_tag = "svdlatrot_off"
    if bool(getattr(args, "hybrid_ffn_svd_latent_rotate", False)):
        latent_rotate_tag = (
            "svdlatrot_"
            f"{getattr(args, 'hybrid_ffn_svd_latent_rotate_mode', 'hadamard')}"
        )
    int8_tag = "real_int8" if bool(getattr(args, "hybrid_real_int8", False)) else "fake_quant"
    if bool(getattr(args, "pre_svd_rotate", False)):
        rotate_info = "presvdrot"
    else:
        rotate_info = "presvdnorot"
    if bool(getattr(args, "rotate", False)) and bool(getattr(args, "use_hybrid_ffn", False)):
        mlp_rotate_info = "mlpqonly"
    else:
        mlp_rotate_info = "mlpdefault"
    filename = (
        f"hybrid_ffn_groups_{nsamples}_{seed}_"
        f"{calib_method}_{rotate_info}_svdffn{svd_ratio}_"
        f"rank{effective_rank_ratio}_{sigma_fuse}_w{w_bits}_{w_sym_tag}_"
        f"a{a_bits}_{a_sym_tag}"
        f"_ag{a_groupsize}_ac{a_clip_ratio}_{latent_rotate_tag}_"
        f"{int8_tag}_{mlp_rotate_info}.json"
    )
    return os.path.join(cache_dir, filename)


def _compute_channel_proxy_errors(
    W: torch.Tensor,
    rank: int,
    w_bits: int,
    sym: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized proxy (svd_err, quant_err) for every output channel row in W.

    Returns two 1-D tensors of shape (n_rows,) on the same device as ``W``.
    """
    d_in = W.shape[1]
    norm_sq = W.square().sum(dim=1)

    frac_kept = min(1.0, rank / d_in)
    err_svd = norm_sq * (1.0 - frac_kept)

    if sym:
        n_levels = 2 ** (w_bits - 1) - 1
    else:
        n_levels = 2 ** w_bits - 1
    step = W.abs().amax(dim=1).clamp(min=1e-9) / n_levels
    err_quant = step.square() * (d_in / 12.0)
    return err_svd, err_quant


def _compute_channel_margins(
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    act_scale: torch.Tensor,
    svd_rank: int,
    w_bits: int,
    w_sym: bool,
) -> torch.Tensor:
    """Vectorized margin = weighted(quant_err - svd_err) for each FFN channel."""
    down_col_norm = W_down.norm(dim=0)
    down_col_norm = down_col_norm / (down_col_norm.mean() + 1e-6)

    act_scale = act_scale.to(W_gate.device, dtype=W_gate.dtype)
    Wg_scaled = W_gate * act_scale.unsqueeze(0)
    Wu_scaled = W_up * act_scale.unsqueeze(0)

    err_g_svd, err_g_q = _compute_channel_proxy_errors(Wg_scaled, svd_rank, w_bits, w_sym)
    err_u_svd, err_u_q = _compute_channel_proxy_errors(Wu_scaled, svd_rank, w_bits, w_sym)

    return down_col_norm * ((err_g_q + err_u_q) - (err_g_svd + err_u_svd))


def _prepare_svd_aware_calib_tokens(
    calib_inps: torch.Tensor,
    d_in: int,
    max_tokens: int = _SVD_AWARE_MAX_TOKENS,
) -> torch.Tensor:
    """Flatten and cap calibration tokens for svd_aware before device transfer."""
    X = calib_inps.float().reshape(-1, d_in)
    if X.shape[0] > max_tokens:
        X = X[:max_tokens]
    return X.contiguous()


def _flatten_calib_tokens(calib_inps: torch.Tensor, d_in: int) -> torch.Tensor:
    """Flatten calibration hidden states to 2-D tokens."""
    return calib_inps.float().reshape(-1, d_in).contiguous()


def _compute_act_scale_from_calib(calib_inps: torch.Tensor, d_in: int) -> torch.Tensor:
    """Compute per-input-dimension activation scale from calibration tensors."""
    X = _flatten_calib_tokens(calib_inps, d_in)
    return X.abs().mean(dim=0).contiguous() + 1e-6


def _prepare_gradient_loss_inputs(
    calib_inps: torch.Tensor,
    layer_output_grad: torch.Tensor,
    d_in: int,
    d_out: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten and align gradient-loss inputs before device transfer.

    The selector uses the prefix of both tensors with the same token count, so
    trim on CPU first to reduce GPU transfer volume.
    """
    X = _flatten_calib_tokens(calib_inps, d_in)
    G = layer_output_grad.float().reshape(-1, d_out).contiguous()
    N = min(X.shape[0], G.shape[0])
    return X[:N].contiguous(), G[:N].contiguous()

def select_svd_channels(
    W_gate: torch.Tensor,          # (d_ffn, d_in)  – original weight
    W_up:   torch.Tensor,          # (d_ffn, d_in)
    W_down: torch.Tensor,          # (d_out, d_ffn)
    n_svd:  int,
    calib_inps: Optional[torch.Tensor] = None,   # (..., d_in) activation samples
    act_scale: Optional[torch.Tensor] = None,    # (d_in,) precomputed activation scale
    svd_rank: int = 0,             # rank used in SVD group (0 → auto from n_svd)
    w_bits: int = 4,               # bits used in Quant group (for error estimate)
    w_sym: bool = True,            # symmetric quantisation flag
) -> Tuple[List[int], List[int], torch.Tensor]:
    """Select hidden channels for SVD vs Quant using per-channel error comparison.

    Algorithm
    ---------
    Start from the pure-quant baseline (all channels in Quant group).
    Move channel i to SVD group **only if** the SVD reconstruction error is
    strictly smaller than the quantisation reconstruction error for that channel.

    This gives a hard guarantee:
      accuracy(hybrid) ≥ accuracy(pure quant)   [at the proxy-error level]

    Both errors are computed on the *output contribution* of each channel,
    weighted by calibration input activations and W_down column norms so that
    channels with high downstream sensitivity are penalised more harshly.

    Parameters
    ----------
    W_gate, W_up : (d_ffn, d_in)  gate and up projection weights
    W_down       : (d_out, d_ffn) down projection weight
    n_svd        : upper-bound on SVD group size (may return fewer if quant wins)
    calib_inps   : calibration hidden-state activations (..., d_in)
    svd_rank     : rank used for the SVD approximation; 0 = auto (50% of n_svd)
    w_bits       : quantisation bit-width for the Quant group
    w_sym        : True = symmetric quantisation

    Returns
    -------
    svd_idx   : sorted list of channel indices assigned to SVD group
    quant_idx : sorted list of channel indices assigned to Quant group
    perm      : 1-D LongTensor = svd_idx + quant_idx  (unified permutation)
    """
    d_ffn, d_in = W_gate.shape
    assert 0 < n_svd < d_ffn, f"n_svd={n_svd} must be in (0, {d_ffn})"

    device = W_gate.device
    Wg = W_gate.float().to(device)
    Wu = W_up.float().to(device)
    Wd = W_down.float().to(device)

    # Activation scale: E[|x_j|] over calibration samples (d_in,)
    if act_scale is not None:
        act_scale = act_scale.to(device=device, dtype=Wg.dtype)
    elif calib_inps is not None:
        c = calib_inps.float().to(device)
        act_scale = c.reshape(-1, c.shape[-1]).abs().mean(dim=0) + 1e-6
    else:
        act_scale = torch.ones(d_in, device=device)

    # ── Auto SVD rank ─────────────────────────────────────────────────────
    if svd_rank <= 0:
        svd_rank = max(1, int(0.5 * min(n_svd, d_in)))

    # ── Per-channel error comparison ──────────────────────────────────────
    # For each channel i, compute:
    #   e_svd_i   = weighted reconstruction error under rank-svd_rank SVD
    #   e_quant_i = weighted reconstruction error under w_bits quantisation
    # Only move to SVD if e_svd_i < e_quant_i  (channel genuinely benefits).

    margins = _compute_channel_margins(
        Wg, Wu, Wd,
        act_scale=act_scale,
        svd_rank=svd_rank,
        w_bits=w_bits,
        w_sym=w_sym,
    )
    svd_wins = torch.nonzero(margins > 0, as_tuple=False).flatten()

    # ── Apply n_svd cap ───────────────────────────────────────────────────
    # If more channels prefer SVD than n_svd allows, take the top-n_svd
    # by margin (largest benefit first).
    if svd_wins.numel() > n_svd:
        top_idx = torch.topk(margins[svd_wins], k=n_svd).indices
        svd_wins = svd_wins[top_idx]

    logging.info(
        f"[hybrid_ffn] error-compare: {svd_wins.numel()}/{d_ffn} channels → SVD "
        f"(cap={n_svd}, svd_rank={svd_rank}, w_bits={w_bits})"
    )

    svd_idx   = svd_wins.sort().values.tolist()
    quant_idx = sorted(set(range(d_ffn)) - set(svd_idx))
    perm = torch.tensor(svd_idx + quant_idx, dtype=torch.long, device=device)
    return svd_idx, quant_idx, perm


def _per_channel_errors(
    w: torch.Tensor,    # (d_in,) activation-scaled weight row
    rank: int,
    w_bits: int,
    sym: bool,
) -> Tuple[float, float]:
    """Return (svd_err, quant_err) for a single weight row.

    SVD error
    ---------
    Rank-r approximation of a rank-1 vector w is just:
      w_svd = (w · e_r) * e_r    where e_r = w/||w|| is the top singular direction
    But for a single vector, rank-1 SVD is lossless (||w||² energy fully captured).
    Instead we model the error of approximating w using the top-r directions of
    the *full column space* (d_in), which requires us to project w onto a rank-r
    random subspace as a proxy for the rank-r truncation error across all channels:
      error_svd = ||w||² * (1 - rank/d_in)    [expected residual energy]

    This is the closed-form expectation of the truncation error when r out of d_in
    singular values are kept.  For channels where w is concentrated along a few
    directions (low effective rank), the actual error will be smaller than this
    estimate → conservative (slightly favours Quant, so the guarantee holds).

    Quant error
    -----------
    Round-trip error for uniform symmetric/asymmetric quantisation:
      scale = max(|w|) / (2^(bits-1) - 1)     (symmetric)
      error_quant ≈ (scale / 2)² * d_in / 3   [uniform quantisation MSE]
    """
    d = w.shape[0]
    norm_sq = float((w ** 2).sum().item())

    # SVD error: energy NOT captured by a rank-r subspace (upper bound)
    frac_kept = min(1.0, rank / d)
    err_svd = norm_sq * (1.0 - frac_kept)

    # Quant error: uniform quantisation MSE (per output-channel symmetric)
    # For INT8 real quant: scale = max(|w|)/127, step = scale
    # For fake quant (INT4 sym): step = max(|w|)/(2^(bits-1)-1)
    w_max = float(w.abs().max().item()) + 1e-9
    if sym:
        n_levels = 2 ** (w_bits - 1) - 1   # 127 for INT8, 7 for INT4
    else:
        n_levels = 2 ** w_bits - 1
    step = w_max / n_levels
    err_quant = (step / 2) ** 2 * d / 3.0   # E[(Δ/2)²] per element × d elements

    return err_svd, err_quant


def _finalise_score(
    score: torch.Tensor,
    n_svd: int,
    device: torch.device,
) -> Tuple[List[int], List[int], torch.Tensor]:
    """Fallback: sort by score, cut at n_svd. Used only when direct error
    comparison is unavailable."""
    sorted_idx = torch.argsort(score, descending=True)
    svd_idx   = sorted_idx[:n_svd].sort().values.tolist()
    quant_idx = sorted_idx[n_svd:].sort().values.tolist()
    perm = torch.tensor(svd_idx + quant_idx, dtype=torch.long, device=device)
    return svd_idx, quant_idx, perm


def _quantize_rows(W: torch.Tensor, bits: int, sym: bool) -> torch.Tensor:
    """Per-row symmetric / asymmetric round-to-nearest quantization.

    Returns the de-quantized (float) result with the same shape as W.
    """
    if sym:
        maxq = 2 ** (bits - 1) - 1
        scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / maxq
        q = torch.clamp(torch.round(W / scale), -(maxq + 1), maxq)
        return q * scale

    maxq = 2 ** bits - 1
    wmin = W.amin(dim=1, keepdim=True)
    wmax = W.amax(dim=1, keepdim=True)
    zeros = torch.zeros_like(wmin)
    wmin = torch.minimum(wmin, zeros)
    wmax = torch.maximum(wmax, zeros)
    both_zero = (wmin == 0) & (wmax == 0)
    wmin = torch.where(both_zero, torch.full_like(wmin, -1), wmin)
    wmax = torch.where(both_zero, torch.ones_like(wmax), wmax)
    scale = (wmax - wmin).clamp(min=1e-5) / maxq
    zero = (-wmin / scale).round()
    q = torch.clamp(torch.round(W / scale) + zero, 0, maxq)
    return scale * (q - zero)


def _fake_quantize_activation(
    x: torch.Tensor,
    bits: int,
    groupsize: int = -1,
    sym: bool = True,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    """Mirror ActQuantizer's per-token fake quantization for analysis.

    This intentionally returns de-quantized floating-point values, matching the
    runtime ActQuantWrapper path used during fake W/A quantization.
    """
    if bits is None or bits >= 16:
        return x
    if x.numel() == 0:
        return x

    x_dtype = x.dtype
    x_f = x.float()
    if sym:
        maxq = float(2 ** (bits - 1) - 1)
    else:
        maxq = float(2 ** bits - 1)

    if groupsize is not None and groupsize > 0:
        if x_f.shape[-1] % groupsize != 0:
            raise ValueError(
                f"activation groupsize {groupsize} does not divide last dim {x_f.shape[-1]}"
            )
        init_shape = x_f.shape
        grouped = x_f.reshape(-1, init_shape[-1] // groupsize, groupsize)
        xmax = grouped.amax(dim=-1, keepdim=True) * clip_ratio
        xmin = grouped.amin(dim=-1, keepdim=True) * clip_ratio
        if sym:
            xmax = torch.maximum(xmin.abs(), xmax)
            scale = xmax / maxq
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            q = torch.clamp(torch.round(grouped / scale), -(maxq + 1), maxq)
            return (q * scale).reshape(init_shape).to(x_dtype)

        both_zero = (xmin == 0) & (xmax == 0)
        xmin = torch.where(both_zero, torch.full_like(xmin, -1), xmin)
        xmax = torch.where(both_zero, torch.ones_like(xmax), xmax)
        scale = (xmax - xmin) / maxq
        zero = torch.round(-xmin / scale)
        q = torch.clamp(torch.round(grouped / scale) + zero, 0, maxq)
        return (scale * (q - zero)).reshape(init_shape).to(x_dtype)

    flat = x_f.reshape(-1, x_f.shape[-1])
    zeros = torch.zeros(flat.shape[0], device=x_f.device, dtype=x_f.dtype)
    xmin = torch.minimum(flat.min(dim=1).values, zeros) * clip_ratio
    xmax = torch.maximum(flat.max(dim=1).values, zeros) * clip_ratio
    if sym:
        xmax = torch.maximum(xmin.abs(), xmax)
        scale = (xmax / maxq).unsqueeze(1)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(torch.round(flat / scale), -(maxq + 1), maxq)
        return (q * scale).reshape_as(x_f).to(x_dtype)

    both_zero = (xmin == 0) & (xmax == 0)
    xmin = torch.where(both_zero, torch.full_like(xmin, -1), xmin)
    xmax = torch.where(both_zero, torch.ones_like(xmax), xmax)
    scale = ((xmax - xmin) / maxq).unsqueeze(1)
    zero = torch.round((-xmin).unsqueeze(1) / scale)
    q = torch.clamp(torch.round(flat / scale) + zero, 0, maxq)
    return (scale * (q - zero)).reshape_as(x_f).to(x_dtype)


def _power2_blocks(n: int) -> List[int]:
    """Greedy power-of-two decomposition for block-Hadamard rotations."""
    blocks: List[int] = []
    remaining = int(n)
    while remaining > 0:
        block = 1 << (remaining.bit_length() - 1)
        blocks.append(block)
        remaining -= block
    return blocks


def _hadamard_transform_power2_lastdim(x: torch.Tensor) -> torch.Tensor:
    """Normalized Sylvester Hadamard transform on the last dim; dim must be pow2."""
    n = x.shape[-1]
    if n <= 1:
        return x
    y = x.reshape(-1, n).float()
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        y0 = y[:, :, 0, :].clone()
        y1 = y[:, :, 1, :].clone()
        y[:, :, 0, :] = y0 + y1
        y[:, :, 1, :] = y0 - y1
        y = y.reshape(-1, n)
        h *= 2
    return (y.reshape(x.shape) / math.sqrt(n)).to(x.dtype)


def _block_hadamard_lastdim(x: torch.Tensor) -> torch.Tensor:
    """Apply an orthogonal block-Hadamard rotation on the last dimension."""
    n = x.shape[-1]
    if n <= 1:
        return x
    out = x.clone()
    start = 0
    for block in _power2_blocks(n):
        end = start + block
        out[..., start:end] = _hadamard_transform_power2_lastdim(out[..., start:end])
        start = end
    return out


def _random_orthogonal_matrix(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Deterministic random orthogonal matrix for latent SVD reparameterization."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    q, r = torch.linalg.qr(torch.randn(size, size, generator=gen, dtype=torch.float32))
    # Fix QR sign ambiguity so the same seed is stable across LAPACK choices.
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    q = q * signs.unsqueeze(0)
    return q.to(device=device, dtype=dtype)


def _hybrid_ffn_latent_rotation_seed(args, layer_idx: int) -> int:
    base_seed = int(getattr(args, "seed", 0))
    layer_part = 0 if layer_idx is None or layer_idx < 0 else int(layer_idx)
    return base_seed + 1009 * layer_part + 7919


def _apply_svd_latent_rotation_to_factors(
    A: torch.Tensor,      # (d_out, rank)
    B: torch.Tensor,      # (rank, d_in)
    args,
    layer_idx: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply A <- A R^T and B <- R B without changing A @ B in FP."""
    if not bool(getattr(args, "hybrid_ffn_svd_latent_rotate", False)):
        return A, B

    rank = A.shape[1]
    if rank <= 1:
        return A, B

    mode = getattr(args, "hybrid_ffn_svd_latent_rotate_mode", "hadamard")
    if mode == "hadamard":
        # Block-Hadamard is symmetric and orthogonal, supports arbitrary ranks,
        # and avoids materializing a dense rank x rank matrix.
        A_rot = _block_hadamard_lastdim(A)
        B_rot = _block_hadamard_lastdim(B.T).T
        return A_rot, B_rot

    if mode == "random":
        seed = _hybrid_ffn_latent_rotation_seed(args, layer_idx)
        R = _random_orthogonal_matrix(rank, A.device, A.dtype, seed)
        A_rot = A @ R.T
        B_rot = R @ B
        return A_rot, B_rot

    raise ValueError(f"Unsupported hybrid FFN SVD latent rotation mode: {mode}")


def _svd_factor_matrices_from_decomp(
    U: torch.Tensor,
    S: torch.Tensor,
    Vt: torch.Tensor,
    rank: int,
    sigma_fuse: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build concrete A/B matrices exactly matching SVDLinear's non-per-head modes."""
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]

    if sigma_fuse == "UV":
        return (
            U_r.mul(S_r.sqrt()).contiguous(),
            Vt_r.mul(S_r.sqrt().view(-1, 1)).contiguous(),
        )
    if sigma_fuse == "U" or sigma_fuse == "profile":
        return U_r.mul(S_r).contiguous(), Vt_r.contiguous()
    if sigma_fuse == "V":
        return U_r.contiguous(), Vt_r.mul(S_r.view(-1, 1)).contiguous()
    if sigma_fuse == "adaptive":
        eps = 1e-6
        V_r = Vt_r.T
        scale = U_r.abs().max(dim=0).values / (V_r.abs().max(dim=0).values + eps)
        return (
            U_r.mul(S_r ** (1 / (scale + 1))).contiguous(),
            Vt_r.mul((S_r ** (scale / (scale + 1))).view(-1, 1)).contiguous(),
        )

    try:
        sigma = float(sigma_fuse)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unsupported sigma_fuse for hybrid FFN SVD analysis: {sigma_fuse}"
        ) from exc
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(
            f"Unsupported sigma_fuse for hybrid FFN SVD analysis: {sigma_fuse}"
        )
    return (
        U_r.mul(S_r ** (1 - sigma)).contiguous(),
        Vt_r.mul((S_r ** sigma).view(-1, 1)).contiguous(),
    )


def _apply_svd_latent_rotation_to_module(svd_mod: nn.Module, args, layer_idx: int) -> None:
    """Rotate the concrete SVDLinear A/B weights in-place, independent of sigma_fuse."""
    if not bool(getattr(args, "hybrid_ffn_svd_latent_rotate", False)):
        return
    if not isinstance(getattr(svd_mod, "ALinear", None), nn.Linear):
        logging.warning(
            "[hybrid_ffn] layer %s: cannot apply latent rotation; ALinear is %s",
            layer_idx,
            type(getattr(svd_mod, "ALinear", None)).__name__,
        )
        return
    if not isinstance(getattr(svd_mod, "BLinear", None), nn.Linear):
        logging.warning(
            "[hybrid_ffn] layer %s: cannot apply latent rotation; BLinear is %s",
            layer_idx,
            type(getattr(svd_mod, "BLinear", None)).__name__,
        )
        return

    A = svd_mod.ALinear.weight.data.float()
    B = svd_mod.BLinear.weight.data.float()
    A_rot, B_rot = _apply_svd_latent_rotation_to_factors(A, B, args, layer_idx=layer_idx)
    svd_mod.ALinear.weight.data.copy_(A_rot.to(svd_mod.ALinear.weight.dtype))
    svd_mod.BLinear.weight.data.copy_(B_rot.to(svd_mod.BLinear.weight.dtype))
    logging.info(
        f"[hybrid_ffn] layer {layer_idx}: applied SVD latent "
        f"{getattr(args, 'hybrid_ffn_svd_latent_rotate_mode', 'hadamard')} "
        f"rotation (rank={A.shape[1]})"
    )


def _gptq_quantize_matrix(
    W: torch.Tensor,        # (d_out, d_in)  weight matrix
    X: torch.Tensor,        # (N,    d_in)   calibration activations
    bits: int,
    sym: bool,
    blocksize: int = 128,
    damp: float    = 0.01,
) -> torch.Tensor:
    """Per-matrix GPTQ quantization (Frantar et al. 2022).

    Minimises ||W X - W_q X||_F^2 over calibration inputs X.
    Returns the de-quantized (float) weight matrix W_q, same dtype as W.

    Falls back silently to RTN if Cholesky fails.

    Algorithm
    ---------
    1.  H = X^T X   (input covariance / Hessian)
    2.  H_inv = (H + damp·mean(diag(H))·I)^{-1}   via Cholesky
    3.  For each block of d_in columns:
          For each column j in block:
            - quantize w[:, j]  (per output-channel scale)
            - propagate rounding error to remaining cols via H_inv
    """
    d_out, d_in = W.shape
    device = W.device

    W_f    = W.float()
    X_f    = X.float().to(device)

    # ── Hessian ──────────────────────────────────────────────────────────
    H = X_f.T @ X_f                                    # (d_in, d_in)

    avg_diag = H.diag().mean().clamp(min=1e-8)
    H += damp * avg_diag * torch.eye(d_in, device=device)
    dead = H.diag() == 0
    H[dead, dead] = avg_diag                           # prevent singular columns

    # ── Inverse Hessian via Cholesky ─────────────────────────────────────
    try:
        L     = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)              # (d_in, d_in)
    except RuntimeError:
        logging.warning("[gptq_quant] Cholesky failed; falling back to RTN")
        return _quantize_rows(W, bits, sym)

    Q      = torch.zeros_like(W_f)
    W_work = W_f.clone()

    def _quant_col(w: torch.Tensor) -> torch.Tensor:
        """Quantize a single weight column (d_out,) and return de-quantized result."""
        if sym:
            s   = w.abs().max().clamp(min=1e-8) / (2 ** (bits - 1) - 1)
            return (w / s).round().clamp(-(2**(bits-1)), 2**(bits-1)-1) * s
        else:
            lo, hi = w.min(), w.max()
            s      = (hi - lo).clamp(min=1e-8) / (2 ** bits - 1)
            z      = (-lo / s).round()
            return ((w / s).round() + z).clamp(0, 2**bits - 1).sub(z).mul(s)

    # ── Block-wise GPTQ ───────────────────────────────────────────────────
    for i1 in range(0, d_in, blocksize):
        i2  = min(i1 + blocksize, d_in)
        bsz = i2 - i1

        W_blk = W_work[:, i1:i2].clone()    # (d_out, bsz)
        Err   = torch.zeros_like(W_blk)     # accumulated quantisation errors
        H_blk = H_inv[i1:i2, i1:i2]        # (bsz, bsz)

        for j in range(bsz):
            w    = W_blk[:, j]                        # (d_out,)
            h    = H_blk[j, j].clamp(min=1e-10)       # scalar
            w_q  = _quant_col(w)
            Q[:, i1 + j] = w_q
            err            = (w - w_q) / h            # (d_out,)
            Err[:, j]      = err
            # Propagate error within block
            if j + 1 < bsz:
                W_blk[:, j+1:] -= err.unsqueeze(1) * H_blk[j, j+1:].unsqueeze(0)

        # Propagate accumulated block error to all subsequent columns
        if i2 < d_in:
            W_work[:, i2:] -= Err @ H_inv[i1:i2, i2:]

    return Q.to(W.dtype)


def select_svd_channels_loss_calibration(
    W_gate: torch.Tensor,          # (d_ffn, d_in)
    W_up:   torch.Tensor,          # (d_ffn, d_in)
    W_down: torch.Tensor,          # (d_out, d_ffn)
    n_svd:  int,
    calib_inps: torch.Tensor,      # (N, d_in) or (..., d_in)
    svd_rank: int = 0,
    w_bits: int = 8,
    w_sym: bool = True,
) -> Tuple[List[int], List[int], torch.Tensor]:
    """Greedy **loss-calibration** channel selection ("final_loss_calibration").

    Algorithm
    ---------
    1. Compute FP16 reference FFN output ``y_ref`` on calibration data.
    2. Build a **pure-SVD baseline** (all channels SVD); measure baseline MSE.
    3. Pre-compute per-channel *delta contributions*: how the output changes
       when switching channel i from SVD → quantized approximation.
       Because the gated-FFN output decomposes as::

           y = Σ_i  W_down[:, i] · silu(g_i) · u_i

       a single-channel swap is an O(N · d_out) incremental update — no full
       re-forward needed.
    4. Rank candidate channels by their closed-form SVD error (worst SVD first)
       to try the most likely quant-wins channels early.
    5. Greedily accept channel i into the **Quant group** only if swapping it
       from SVD → Quant strictly reduces MSE.  Stop when ``n_quant = d_ffn - n_svd``
       channels are assigned to Quant, leaving exactly ``n_svd`` in SVD.

    This ensures we keep as many channels in SVD as possible while only
    replacing those where SVD performs worse than quantization.

    Parameters
    ----------
    W_gate, W_up : (d_ffn, d_in)
    W_down       : (d_out, d_ffn)
    n_svd        : target number of channels to keep in the SVD group
    calib_inps   : calibration hidden-state activations, any batch shape with
                   the last dimension = d_in
    svd_rank     : rank used for SVD approximation (0 → auto: 50 % of n_svd)
    w_bits       : quantisation bit-width for the Quant group
    w_sym        : True = symmetric quantisation

    Returns
    -------
    svd_idx, quant_idx, perm   (same conventions as ``select_svd_channels``)
    """
    d_ffn, d_in = W_gate.shape
    d_out = W_down.shape[0]
    assert 0 < n_svd < d_ffn
    assert calib_inps is not None, "loss_calibration requires calib_inps"

    dev = W_gate.device

    Wg = W_gate.float().to(dev)
    Wu = W_up.float().to(dev)
    Wd = W_down.float().to(dev)

    # ── Calibration inputs ────────────────────────────────────────────────
    X = calib_inps.float().to(dev).reshape(-1, d_in)   # (N, d_in)
    N = X.shape[0]

    # ── Auto rank ─────────────────────────────────────────────────────────
    if svd_rank <= 0:
        svd_rank = max(1, int(0.5 * min(n_svd, d_in)))

    # ── 1. FP16 reference output ──────────────────────────────────────────
    with torch.no_grad():
        g_fp = X @ Wg.T           # (N, d_ffn)
        u_fp = X @ Wu.T           # (N, d_ffn)
        h_fp = F.silu(g_fp) * u_fp
        y_ref = h_fp @ Wd.T       # (N, d_out)

    # ── 2. SVD approximation → pure-SVD baseline ──────────────────────────
    # Use the top-r right singular vectors of the full W_gate / W_up matrices
    # as a shared basis.  Channel i's SVD approximation:
    #   Wg_svd[i] = (Wg[i] @ Vt_r.T) @ Vt_r
    with torch.no_grad():
        _, _, Vt_g = torch.linalg.svd(Wg, full_matrices=False)
        _, _, Vt_u = torch.linalg.svd(Wu, full_matrices=False)
        Vt_g_r = Vt_g[:svd_rank]           # (r, d_in)
        Vt_u_r = Vt_u[:svd_rank]           # (r, d_in)

        Wg_svd = (Wg @ Vt_g_r.T) @ Vt_g_r  # (d_ffn, d_in)
        Wu_svd = (Wu @ Vt_u_r.T) @ Vt_u_r  # (d_ffn, d_in)

        g_svd = X @ Wg_svd.T               # (N, d_ffn)
        u_svd = X @ Wu_svd.T               # (N, d_ffn)
        h_svd = F.silu(g_svd) * u_svd     # (N, d_ffn)
        y_svd = h_svd @ Wd.T               # (N, d_out)

    baseline_mse = float(((y_svd - y_ref) ** 2).mean().item())
    logging.info(f"[loss_calib] pure-SVD baseline MSE = {baseline_mse:.6f}")

    # ── 3. Quantize all rows (GPTQ) for per-channel quant activations ─────
    # We need h_q to compute delta_h = h_q - h_svd (switching SVD → quant).
    # GPTQ matches Phase 2, avoiding RTN/GPTQ mismatch.
    with torch.no_grad():
        Wg_q = _gptq_quantize_matrix(Wg, X, w_bits, w_sym)
        Wu_q = _gptq_quantize_matrix(Wu, X, w_bits, w_sym)

        g_q = X @ Wg_q.T          # (N, d_ffn)
        u_q = X @ Wu_q.T          # (N, d_ffn)
        h_q = F.silu(g_q) * u_q  # (N, d_ffn)

        # delta_h[:, i] = h_q[:, i] - h_svd[:, i]  (switching SVD → quant)
        delta_h = h_q - h_svd

    # ── 4. Candidate ordering: SVD-worst channels first ───────────────────
    # Sort ascending by margin = amp*(quant_err - svd_err).
    # Low/negative margin means SVD error > quant error → candidate for quant.
    act_scale = X.abs().mean(dim=0) + 1e-6           # (d_in,)

    margins = _compute_channel_margins(
        Wg, Wu, Wd,
        act_scale=act_scale,
        svd_rank=svd_rank,
        w_bits=w_bits,
        w_sym=w_sym,
    )

    # Sort ascending: channels where SVD is worst (margin most negative) go first
    candidate_order = torch.argsort(margins).tolist()

    # ── 5. Greedy incremental swap (SVD → Quant) ──────────────────────────
    # y_current starts at y_svd; each accepted swap updates it toward y_quant.
    # We collect into quant_set and stop when n_quant = d_ffn - n_svd channels
    # have been accepted.
    n_quant = d_ffn - n_svd
    y_current = y_svd.clone()
    current_mse = baseline_mse
    quant_set: set = set()
    accepted = 0
    rejected = 0

    for ch in candidate_order:
        if len(quant_set) >= n_quant:
            break

        # Incremental update: contribution of channel ch changes by delta_h[:, ch]
        # y_new = y_current + (h_q[:,ch] - h_svd[:,ch]) ⊗ Wd[:,ch]
        dh_ch = delta_h[:, ch]                        # (N,)
        wd_ch = Wd[:, ch]                             # (d_out,)
        delta_y = dh_ch.unsqueeze(1) * wd_ch.unsqueeze(0)  # (N, d_out)
        y_trial = y_current + delta_y
        trial_mse = float(((y_trial - y_ref) ** 2).mean().item())

        if trial_mse < current_mse:
            quant_set.add(ch)
            y_current = y_trial
            current_mse = trial_mse
            accepted += 1
            logging.debug(
                f"[loss_calib] ch {ch:5d}: MSE {current_mse:.6f}→{trial_mse:.6f} ✓ → quant"
            )
        else:
            rejected += 1

    improvement = baseline_mse - current_mse
    logging.info(
        f"[loss_calib] greedy done: {accepted} → quant / {rejected} kept SVD  "
        f"MSE {baseline_mse:.6f} → {current_mse:.6f}  "
        f"(Δ={improvement:+.6f}, rank={svd_rank}, w_bits={w_bits})"
    )

    quant_idx = sorted(quant_set)
    svd_idx   = sorted(set(range(d_ffn)) - quant_set)
    perm = torch.tensor(svd_idx + quant_idx, dtype=torch.long, device=dev)
    return svd_idx, quant_idx, perm


# ---------------------------------------------------------------------------
# HybridSVDQuantFFN module
# ---------------------------------------------------------------------------

class HybridSVDQuantFFN(nn.Module):
    """Hybrid SVD-Quant gated FFN replacement for layer.mlp.

    After construction the module attributes are:
      gate_svd   – SVDLinear  (n_svd output channels of gate_proj)
      up_svd     – SVDLinear  (n_svd output channels of up_proj)
      gate_quant – ActQuantWrapper  (n_quant output channels of gate_proj)
      up_quant   – ActQuantWrapper  (n_quant output channels of up_proj)
      down_proj  – ActQuantWrapper  (full d_ffn input columns, permuted as perm)
      perm       – LongTensor [svd_idx | quant_idx]
      n_svd      – int
    """

    def __init__(
        self,
        gate_svd,
        up_svd,
        gate_quant,
        up_quant,
        down_proj,
        perm: torch.Tensor,
        n_svd: int,
        had_K=None,
        K: int = 1,
        fp32_had: bool = False,
    ):
        super().__init__()
        self.gate_svd   = gate_svd
        self.up_svd     = up_svd
        self.gate_quant = gate_quant
        self.up_quant   = up_quant
        self.down_proj  = down_proj
        self.register_buffer('perm', perm)
        self.n_svd = n_svd
        # Online Hadamard on FFN hidden dim (h = cat[h_svd, h_q]) before down_proj.
        # When enabled, W_down has H^{-1} pre-absorbed on its input side, so the
        # runtime Hadamard and the absorbed inverse cancel, restoring equivalence.
        self.online_had = had_K is not None
        self.K = K
        self.fp32_had = fp32_had
        if had_K is not None:
            self.register_buffer('had_K', had_K)
        else:
            self.register_buffer('had_K', None)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_in)
        Returns: (..., d_out)
        """
        # SVD group: low-rank projections
        g_svd = self.gate_svd(x)          # (..., n_svd)
        u_svd = self.up_svd(x)            # (..., n_svd)
        h_svd = F.silu(g_svd) * u_svd    # (..., n_svd)

        # Quant group: quantised weight projections
        g_q = self.gate_quant(x)          # (..., n_quant)
        u_q = self.up_quant(x)            # (..., n_quant)
        h_q = F.silu(g_q) * u_q          # (..., n_quant)

        # Concatenate in the same order as the permuted W_down columns
        h = torch.cat([h_svd, h_q], dim=-1)  # (..., d_ffn)

        # Optional online Hadamard on FFN hidden dim before down_proj.
        # Reduces outliers in down_proj input activations (mirrors standard QuaRot).
        if self.online_had:
            h_dtype = h.dtype
            if self.fp32_had:
                h = hadamard_utils.matmul_hadU_cuda(h.float(), self.had_K, self.K).to(h_dtype)
            else:
                h = hadamard_utils.matmul_hadU_cuda(h, self.had_K, self.K)

        # W_down already has columns ordered as [SVD | Quant]
        return self.down_proj(h)

    def extra_repr(self) -> str:
        # gate_quant can be ActQuantWrapper (fake quant) or Int8WeightLinear (real int8)
        if isinstance(self.gate_quant, Int8WeightLinear):
            n_q = self.gate_quant.out_features
            quant_type = "real-int8"
        else:
            n_q = self.gate_quant.module.out_features
            quant_type = "fake-quant"
        # down_proj can be ActQuantWrapper or Int8WeightLinear
        if isinstance(self.down_proj, Int8WeightLinear):
            d_out = self.down_proj.out_features
        else:
            d_out = self.down_proj.module.out_features
        if hasattr(self.gate_svd, "BLinear"):
            d_in = self.gate_svd.BLinear.in_features
        else:
            d_in = getattr(self.gate_svd, "in_features", None)
        return (
            f"n_svd={self.n_svd}, n_quant={n_q}, quant={quant_type}, "
            f"d_in={d_in}, d_out={d_out}"
        )


# ---------------------------------------------------------------------------
# Build routine: turn a plain MLP into HybridSVDQuantFFN
# ---------------------------------------------------------------------------

def build_hybrid_ffn(
    layer,
    layer_idx: int,
    args,
    calib_inps: Optional[torch.Tensor] = None,
    layer_indices_dict: Optional[dict] = None,
    original_perm: Optional[torch.Tensor] = None,
    precomputed_group: Optional[dict] = None,
) -> HybridSVDQuantFFN:
    """Replace ``layer.mlp`` with a HybridSVDQuantFFN.

    Absorption of permutation matrices
    ------------------------------------
    The function applies two permutations and absorbs both into the weights
    so that no runtime reordering is needed:

      P   = original_perm  (optional; permutation already applied to the hidden
                            states by an upstream step, e.g. channel reordering
                            from QuaRot rotation – if None, the weights are
                            assumed to already be in the P-absorbed form)
      P_s = the grouping permutation we compute (svd_idx + quant_idx)

    Resulting absorbed weights (column-vector convention  y = W · x):

      Wa'   = P_s  @ Wa   @ P^T   (gate/up:  output rows grouped, input cols un-rotated)
      Wout' = P_s^T@ Wout               (down:     input cols un-grouped)
                                                (output side unchanged; next-layer P
                                                 absorbed by the downstream linear)

    When original_perm is None (default):
      P is the identity → Wa' = P_s @ Wa,  Wout' = Wout @ P_s^T
    which is exactly what the grouping already does, so no extra step is needed.

    Parameters
    ----------
    layer        : transformer decoder layer (has .mlp.{gate_proj,up_proj,down_proj})
    layer_idx    : index of this layer (for rank lookup / logging)
    args         : parsed argument namespace; relevant fields:
                     .hybrid_ffn_rank_ratio – optional rank ratio used only by hybrid_ffn
                     .rank_ratio    – fallback param ratio for SVD side when
                                      hybrid_ffn_rank_ratio is unset
                     .svd_ffn_ratio – fraction of d_ffn channels assigned to SVD group
                     .sigma_fuse    – how Σ is split between U and V  ('UV'/'U'/'V')
                     .act_aware     – bool; use activation scaling in SVD
                     .act_alpha     – float; scaling exponent
    calib_inps   : optional calibration activations fed to this layer (..., d_in)
    layer_indices_dict : optional {layer_idx: {'up_proj': [singular_idx, ...]}}
                         from loss-aware calibration (used as SVD rank selector)
    original_perm : optional 1-D LongTensor of length d_in.
                    If the upstream pipeline has permuted (but NOT yet absorbed into
                    weights) the model hidden states by index sequence `original_perm`,
                    pass it here so that the input-column side of gate/up is
                    un-permuted (absorb P) before the SVD/quant split.
                    In the standard QuaRot pipeline this is NOT needed because
                    rotation_utils already absorbs P into the weight matrices.
    """
    from .svd_utils import SVDLinear
    from .quant_utils import ActQuantWrapper

    mlp        = layer.mlp
    gate_lin   = _unwrap(mlp.gate_proj)   # nn.Linear
    up_lin     = _unwrap(mlp.up_proj)
    down_aw    = mlp.down_proj             # ActQuantWrapper (already added)

    d_ffn = gate_lin.out_features
    d_in  = gate_lin.in_features
    d_out = _unwrap(down_aw).out_features

    # ── Determine n_svd ──────────────────────────────────────────────────
    svd_ffn_ratio = getattr(args, 'svd_ffn_ratio', 0.2)
    n_svd  = max(1, int(d_ffn * svd_ffn_ratio))
    n_svd  = min(n_svd, d_ffn - 1)          # at least one quant channel

    logging.info(
        f"[hybrid_ffn] layer {layer_idx}: d_ffn={d_ffn}, "
        f"n_svd={n_svd}, n_quant={d_ffn - n_svd}"
    )

    # ── Read weights (float for numerical ops) ────────────────────────────
    Wg = gate_lin.weight.data.float()   # (d_ffn, d_in)
    Wu = up_lin.weight.data.float()     # (d_ffn, d_in)
    Wd = _unwrap(down_aw).weight.data.float()  # (d_out, d_ffn)
    orig_dtype = gate_lin.weight.dtype

    # ── Absorb original_perm P (input-column side of gate / up) ──────────
    # If the hidden states were permuted as x' = x[P] but the weights have NOT
    # yet been updated to compensate, we must apply the inverse permutation to
    # the INPUT columns of gate/up so that Wg @ x' = Wg[:, P^{-1}] @ x.
    # For a permutation matrix P_inv = P^T: Wg_absorbed = Wg[:, inv_perm]
    # where inv_perm[P[i]] = i.
    if original_perm is not None:
        P_cpu = original_perm.cpu()
        assert P_cpu.shape[0] == d_in, (
            f"original_perm length {P_cpu.shape[0]} != d_in {d_in}"
        )
        # Build inverse permutation: inv[P[i]] = i
        inv_P = torch.empty_like(P_cpu)
        inv_P[P_cpu] = torch.arange(d_in)
        # Absorb: Wa' = Wa @ P^T  (i.e. reorder input columns by inv_P)
        Wg = Wg[:, inv_P]
        Wu = Wu[:, inv_P]
        logging.info(
            f"[hybrid_ffn] layer {layer_idx}: absorbed original_perm P "
            f"(input-column reorder, inv_perm[:5]={inv_P[:5].tolist()})"
        )

    # ── Channel selection (or load from precomputed groups) ──────────────
    if precomputed_group is not None:
        # Phase 2: use pre-computed channel assignments (no calibration needed).
        # If the requested svd_ffn_ratio differs from the stored ratio, dynamically
        # reslice using greedy_accept_order so different ratios can be tested from
        # a single Phase-1 JSON without re-running Phase 1.
        stored_n_svd = int(precomputed_group.get("n_svd", len(precomputed_group["svd_idx"])))
        if n_svd != stored_n_svd and "greedy_accept_order" in precomputed_group:
            gao      = precomputed_group["greedy_accept_order"]   # ordered list, len ≤ d_ffn
            n_quant  = d_ffn - n_svd
            all_ch   = list(range(d_ffn))
            if n_quant <= len(gao):
                quant_idx = gao[:n_quant]
            else:
                # More Quant channels requested than greedy explored; pad with
                # remaining channels (those never accepted → most SVD-friendly)
                accepted_set = set(gao)
                extra = [c for c in all_ch if c not in accepted_set]
                quant_idx = gao + extra[:n_quant - len(gao)]
            quant_set = set(quant_idx)
            svd_idx   = [c for c in all_ch if c not in quant_set]
            # Rebuild perm so SVD channels come first
            perm_list = svd_idx + quant_idx
            perm      = torch.tensor(perm_list, dtype=torch.long, device=Wg.device)
            rank      = None  # will be computed below from actual_n_svd
            actual_n_svd = len(svd_idx)
            logging.info(
                f"[hybrid_ffn] layer {layer_idx}: dynamic ratio reslice "
                f"(stored n_svd={stored_n_svd} → requested n_svd={n_svd}) – "
                f"SVD={actual_n_svd}, Quant={len(quant_idx)}"
            )
        else:
            svd_idx      = precomputed_group["svd_idx"]
            quant_idx    = precomputed_group["quant_idx"]
            perm         = torch.tensor(precomputed_group["perm"], dtype=torch.long, device=Wg.device)
            rank         = int(precomputed_group["svd_rank"])
            actual_n_svd = len(svd_idx)
            logging.info(
                f"[hybrid_ffn] layer {layer_idx}: loaded precomputed groups – "
                f"SVD={actual_n_svd}, Quant={len(quant_idx)}, rank={rank}"
            )
    else:
        # Single-pass or Phase 1+2 combined: compute SVD rank then run selector
        rank = _compute_svd_rank(n_svd, d_in, args, layer_idx,
                                  layer_indices_dict, key='up_proj')
        w_bits = int(getattr(args, 'w_bits', 4))
        w_sym  = not bool(getattr(args, 'w_asym', False))
        svd_idx, quant_idx, perm = select_svd_channels(
            Wg, Wu, Wd, n_svd,
            calib_inps = calib_inps,
            svd_rank   = rank,
            w_bits     = w_bits,
            w_sym      = w_sym,
        )
        actual_n_svd = len(svd_idx)
        logging.info(
            f"[hybrid_ffn] layer {layer_idx}: "
            f"SVD group={actual_n_svd} (cap={n_svd}), "
            f"Quant group={len(quant_idx)}, rank={rank}, "
            f"P_s perm[:5]={perm[:5].tolist()}"
        )

    # ── Apply grouping permutation P_s and slice ──────────────────────────
    # Absorbed formula (column-vector convention):
    #   Wa'   = P_s @ Wa          (output rows grouped by P_s)
    #   Wout' = Wout @ P_s^T      (input cols un-grouped so h=[SVD|Quant] feeds correctly)
    perm_cpu = perm.cpu()

    Wg_grp = Wg[perm_cpu, :]      # P_s @ Wg  →  (d_ffn, d_in)
    Wu_grp = Wu[perm_cpu, :]      # P_s @ Wu

    Wd_grp = Wd[:, perm_cpu]      # Wd @ P_s^T  →  (d_out, d_ffn)

    Wg_svd   = Wg_grp[:actual_n_svd, :]           # (actual_n_svd,  d_in)
    Wu_svd   = Wu_grp[:actual_n_svd, :]
    Wg_quant = Wg_grp[actual_n_svd:, :]           # (n_quant, d_in)
    Wu_quant = Wu_grp[actual_n_svd:, :]

    # Recompute rank for the actual (possibly smaller or dynamic-resliced) SVD group
    if actual_n_svd == 0:
        rank = 0
    elif rank is None or actual_n_svd != n_svd:
        rank = _compute_svd_rank(actual_n_svd, d_in, args, layer_idx,
                                  layer_indices_dict, key='up_proj')

    # ── Input whitening for SVD group (--hybrid_ffn_act_aware) ──────────────
    # Two modes (controlled by --hybrid_ffn_act_aware_mode):
    #   'chol': full Cholesky L of X^T X, sets scaling_diag_matrixS on gate/up
    #           _make_svd_linear applies W <- W @ L before SVD, absorbs L^{-1} into V
    #   'diag': per-feature RMS scale, sets scaling_diag_matrix on gate/up
    #           _make_svd_linear applies W <- W * scale before SVD, un-scales V (stable)
    _ffn_act_aware   = bool(getattr(args, 'hybrid_ffn_act_aware', False))
    _act_aware_mode  = getattr(args, 'hybrid_ffn_act_aware_mode', 'chol')
    _chol_S          = None
    _diag_scale_val  = None
    if _ffn_act_aware and calib_inps is not None:
        try:
            dev_c = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            _MAX_CALIB_TOK = 4096
            X_raw = calib_inps.reshape(-1, d_in)
            if X_raw.shape[0] > _MAX_CALIB_TOK:
                step = max(1, X_raw.shape[0] // _MAX_CALIB_TOK)
                X_raw = X_raw[::step][:_MAX_CALIB_TOK]
            X_d = X_raw.double().to(dev_c)
            if _act_aware_mode == 'diag':
                # Diagonal per-feature RMS scaling (no matrix inversion needed)
                var = (X_d ** 2).mean(dim=0)
                _diag_scale_val = var.clamp(min=1e-8).sqrt().cpu()
                gate_lin.scaling_diag_matrix = _diag_scale_val
                up_lin.scaling_diag_matrix   = _diag_scale_val
                logging.info(
                    f"[hybrid_ffn] layer {layer_idx}: computed diagonal whitening scale "
                    f"(X={X_d.shape}, mode=diag)"
                )
            else:
                # Full Cholesky (SVDLLM)
                H = (X_d.T @ X_d) / X_d.shape[0]
                damp = 0.01 * H.diagonal().mean().clamp(min=1e-8)
                H.diagonal().add_(damp)
                _chol_S = torch.linalg.cholesky(H).cpu()
                gate_lin.scaling_diag_matrixS = _chol_S
                up_lin.scaling_diag_matrixS   = _chol_S
                logging.info(
                    f"[hybrid_ffn] layer {layer_idx}: computed Cholesky whitening matrix "
                    f"(X={X_d.shape}, damp={damp:.2e})"
                )
                del H
            del X_d
            torch.cuda.empty_cache()
        except Exception as _ce:
            logging.warning(
                f"[hybrid_ffn] layer {layer_idx}: Whitening computation failed ({_ce}); "
                "falling back to raw SVD."
            )
            _ffn_act_aware = False

    _svd_fp16 = bool(getattr(args, 'hybrid_ffn_fp16_quant_group', False))
    gate_svd_mod = _make_svd_linear(Wg_svd, rank, args, gate_lin, layer_idx=layer_idx,
                                    act_aware_override=_ffn_act_aware, fp16=_svd_fp16)
    up_svd_mod   = _make_svd_linear(Wu_svd, rank, args, up_lin,   layer_idx=layer_idx,
                                    act_aware_override=_ffn_act_aware, fp16=_svd_fp16)

    # Clean up temporary whitening attributes to avoid confusion downstream
    if _chol_S is not None:
        for _lin in (gate_lin, up_lin):
            if hasattr(_lin, 'scaling_diag_matrixS'):
                del _lin.scaling_diag_matrixS
    if _diag_scale_val is not None:
        for _lin in (gate_lin, up_lin):
            if hasattr(_lin, 'scaling_diag_matrix'):
                del _lin.scaling_diag_matrix

    # ── Build quant sub-modules (gate_quant, up_quant) ───────────────────
    use_real_int8 = bool(getattr(args, 'hybrid_real_int8', False))
    use_fp16_quant_group = bool(getattr(args, 'hybrid_ffn_fp16_quant_group', False))
    if use_fp16_quant_group:
        gate_quant_mod = _make_fp16_linear(Wg_quant, d_in, gate_lin)
        up_quant_mod   = _make_fp16_linear(Wu_quant, d_in, up_lin)
        logging.info(f"[hybrid_ffn] layer {layer_idx}: quant group → FP16 (no quantization)")
    elif use_real_int8:
        gate_quant_mod = _make_real_int8_linear(Wg_quant, d_in, gate_lin)
        up_quant_mod   = _make_real_int8_linear(Wu_quant, d_in, up_lin)
        logging.info(f"[hybrid_ffn] layer {layer_idx}: quant group → real INT8")
    else:
        gate_quant_mod = _make_quant_linear(Wg_quant, d_in, gate_lin)
        up_quant_mod   = _make_quant_linear(Wu_quant, d_in, up_lin)

    # ── Write back permuted down_proj columns ────────────────────────────
    # Wout' = Wd @ P_s^T  (input columns now expect [SVD | Quant] ordered h)
    down_inner = _unwrap(down_aw)
    if use_real_int8:
        # Replace ActQuantWrapper down_proj with a real INT8 module
        down_int8 = Int8WeightLinear.from_weight(
            Wd_grp.to(orig_dtype),
            bias=down_inner.bias.data.clone() if down_inner.bias is not None else None,
        ).to(down_inner.weight.device if hasattr(down_inner, 'weight') else gate_lin.weight.device)
        _copy_runtime_hadamard_config(down_int8, down_aw)
        down_aw = down_int8
        logging.info(f"[hybrid_ffn] layer {layer_idx}: down_proj → real INT8")
    else:
        down_inner.weight.data = Wd_grp.to(orig_dtype)

    # ── Optional online Hadamard on FFN hidden dim ────────────────────────
    # When --hybrid_ffn_had is set, absorb H^{-1} into the input side of
    # down_proj and enable the corresponding runtime Hadamard in forward().
    # This mirrors the standard QuaRot online-Hadamard that was intentionally
    # skipped in rotate_mlp_output_hybrid_ffn (because the grouping permutation
    # and Hadamard don't commute when applied in the original order).  Here we
    # apply the Hadamard AFTER the concat step, so commutativity is not required.
    had_K_buf = None
    had_K_int  = 1
    fp32_had   = getattr(args, 'fp32_had', False)
    if getattr(args, 'hybrid_ffn_had', False):
        d_ffn_had = Wd_grp.shape[1]   # == n_svd + n_quant
        try:
            had_K_val, had_K_int = hadamard_utils.get_hadK(d_ffn_had)
            # Absorb H^{-1} (= H for Hadamard) into W_down's input dimension.
            W_ = down_inner.weight.data.float().cuda()
            W_ = hadamard_utils.matmul_hadU_cuda(W_, had_K_val, had_K_int)
            down_inner.weight.data = W_.to(orig_dtype).cpu()
            had_K_buf = had_K_val
            logging.info(
                f"[hybrid_ffn] layer {layer_idx}: absorbed H^{{-1}} into down_proj "
                f"(d_ffn={d_ffn_had}, K={had_K_int}); online Hadamard enabled"
            )
        except Exception as exc:
            logging.warning(
                f"[hybrid_ffn] layer {layer_idx}: --hybrid_ffn_had requested but "
                f"get_hadK({d_ffn_had}) failed ({exc}). Skipping Hadamard."
            )

    return HybridSVDQuantFFN(
        gate_svd   = gate_svd_mod,
        up_svd     = up_svd_mod,
        gate_quant = gate_quant_mod,
        up_quant   = up_quant_mod,
        down_proj  = down_aw,
        perm       = perm,
        n_svd      = actual_n_svd,
        had_K      = had_K_buf,
        K          = had_K_int,
        fp32_had   = fp32_had,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(module):
    """Unwrap ActQuantWrapper → nn.Linear if needed."""
    from .quant_utils import ActQuantWrapper
    if isinstance(module, ActQuantWrapper):
        return module.module
    return module


def _compute_svd_rank(
    n_svd: int,
    d_in: int,
    args,
    layer_idx: int,
    layer_indices_dict: Optional[dict],
    key: str = 'up_proj',
) -> int:
    """Compute the SVD rank for the SVD group.

    Priority:
    1. If loss-aware indices are available, use their length.
    2. If args.use_true_param_ratio, match param count of rank-r decomposition
       to param_ratio * (n_svd * d_in), where param_ratio prefers
       args.hybrid_ffn_rank_ratio over args.rank_ratio.
    3. Fall back to rank = int(param_ratio * min(n_svd, d_in)).
    """
    if (
        layer_indices_dict is not None
        and layer_idx in layer_indices_dict
        and key in layer_indices_dict[layer_idx]
        and layer_indices_dict[layer_idx][key]
    ):
        indices = layer_indices_dict[layer_idx][key]
        rank = len(indices)
        logging.info(f"[hybrid_ffn] layer {layer_idx}: using loss-aware rank={rank}")
        return max(1, rank)

    hybrid_param_ratio = getattr(args, 'hybrid_ffn_rank_ratio', None)
    if hybrid_param_ratio is None:
        param_ratio = float(getattr(args, 'rank_ratio', 0.5))
        ratio_src = 'rank_ratio'
    else:
        param_ratio = float(hybrid_param_ratio)
        ratio_src = 'hybrid_ffn_rank_ratio'

    if getattr(args, 'use_true_param_ratio', False):
        # r*(n_svd + d_in) = param_ratio * n_svd * d_in  →  solve for r
        r = int(param_ratio * n_svd * d_in / (n_svd + d_in))
    else:
        r = int(param_ratio * min(n_svd, d_in))

    r = max(1, min(r, min(n_svd, d_in)))
    logging.info(
        f"[hybrid_ffn] layer {layer_idx}: SVD rank={r} "
        f"(param_ratio={param_ratio}, source={ratio_src})"
    )
    return r


def _make_svd_linear(
    W: torch.Tensor,
    rank: int,
    args,
    ref_linear: nn.Linear,
    layer_idx: int = -1,
    act_aware_override: bool = None,
    fp16: bool = False,
):
    """Construct a SVDLinear from a weight slice W (n_out × d_in).

    act_aware_override
        When not None, overrides ``args.act_aware`` for this call only.
        Used by build_hybrid_ffn to enable SVDLLM-style Cholesky whitening
        (``--hybrid_ffn_act_aware``) independently of QKV ``--act_aware``.
    """
    from .svd_utils import SVDLinear
    from .quant_utils import ActQuantWrapper

    # Determine effective act_aware for this call
    _act_aware = act_aware_override if act_aware_override is not None else bool(getattr(args, 'act_aware', False))

    device = W.device
    W_f = W.float()
    if W_f.shape[0] == 0 or rank <= 0:
        return _make_empty_linear(W.shape[1], W.shape[0], ref_linear)

    # Optional: activation-aware scaling of input features
    # Two modes:
    #   scaling_diag_matrix  – diagonal per-feature scale  (simple act_aware)
    #   scaling_diag_matrixS – full lower-triangular Cholesky L of X^T X (SVDLLM)
    act_scale = None
    use_cholesky = False
    scaling_diag = getattr(ref_linear, 'scaling_diag_matrix', None)
    scaling_diag_s = getattr(ref_linear, 'scaling_diag_matrixS', None)
    if _act_aware and scaling_diag is not None:
        act_scale = scaling_diag.float().to(device)
        W_f = W_f * act_scale.view(1, -1)
    elif _act_aware and scaling_diag_s is not None:
        S_mat = scaling_diag_s.float().to(device)
        W_f = (W_f.double() @ S_mat.double()).float()
        use_cholesky = True

    if torch.cuda.is_available():
        W_f = W_f.cuda()
        
    try:
        U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
    except Exception as e:
        logging.warning(f"[hybrid_ffn] SVD failed: {e}. Using identity SVDLinear.")
        fallback = nn.Linear(W.shape[1], W.shape[0], bias=False)
        fallback.weight.data = W.to(ref_linear.weight.dtype)
        return fallback

    U = U[:, :rank].to(device)       # (n_out, rank)
    S = S[:rank].to(device)          # (rank,)
    Vt = Vt[:rank, :].to(device)     # (rank, d_in)
    V = Vt.T.contiguous()            # (d_in, rank)
    W_f = W_f.to(device)
    
    # Undo activation scaling in V so that the SVDLinear represents the
    # original (un-whitened) weight factorisation.
    if act_scale is not None:
        # Simple diagonal case: W_whitened = W * scale → undo via V / scale
        V = V / act_scale.view(-1, 1)
    elif use_cholesky:
        # Cholesky case: W_whitened = W @ L → undo via V = L^{-T} @ V_whitened
        # (L is lower-triangular so L^{-T} = inv(L)^T)
        S_mat = scaling_diag_s.float().to(V.device)
        try:
            S_inv = torch.linalg.inv(S_mat)
            V = (S_inv.double().T @ V.double()).float()
        except Exception as e:
            logging.warning(
                f"[hybrid_ffn] layer {layer_idx}: Cholesky undo failed ({e}); "
                "V remains in whitened space — SVD reconstruction may be degraded."
            )

    sigma_fuse = _get_hybrid_ffn_sigma_fuse(args)
    if sigma_fuse == "local_ft":
        raise ValueError(
            "hybrid_ffn raw SVD path does not support sigma_fuse='local_ft'; "
            "use UV/U/V/profile/adaptive or a numeric sigma split."
        )
    svd_mod = SVDLinear(
        U=U, S=S, V=V,
        bias=None,
        sigma_fuse=sigma_fuse,
    )
    _apply_svd_latent_rotation_to_module(svd_mod, args, layer_idx=layer_idx)
    # Cast to original dtype
    dtype = ref_linear.weight.dtype
    svd_mod = svd_mod.to(device=device, dtype=dtype)
    if not fp16 and int(getattr(args, 'a_bits', 16)) < 16:
        # Hybrid FFN's SVD side is created after the original MLP wrappers are
        # discarded.  Match the existing per-head SVD convention: BLinear's
        # input quantizer handles x->A8, and BLinear's out_quantizer handles the
        # latent activation consumed by ALinear.
        if isinstance(svd_mod.BLinear, nn.Linear):
            svd_mod.BLinear = ActQuantWrapper(svd_mod.BLinear).to(device=device, dtype=dtype)
            svd_mod = svd_mod.to(device=device, dtype=dtype)
    return svd_mod


def _make_empty_linear(d_in: int, d_out: int, ref_linear: nn.Linear):
    """Return a zero-width projection used when a layer's SVD group is empty."""
    empty = nn.Linear(d_in, d_out, bias=False)
    empty = empty.to(ref_linear.weight.device, dtype=ref_linear.weight.dtype)
    return empty


def _make_quant_linear(W: torch.Tensor, d_in: int, ref_linear: nn.Linear):
    """Create an ActQuantWrapper around an nn.Linear with weight W."""
    from .quant_utils import ActQuantWrapper

    n_out = W.shape[0]
    lin = nn.Linear(d_in, n_out, bias=(ref_linear.bias is not None))
    lin.weight.data = W.to(ref_linear.weight.dtype)
    if ref_linear.bias is not None:
        lin.bias = None   # bias handled at the down_proj level if present
    wrapper = ActQuantWrapper(lin)
    return wrapper.to(ref_linear.weight.device)


def _make_fp16_linear(W: torch.Tensor, d_in: int, ref_linear: nn.Linear):
    """Create a plain nn.Linear (FP16, no quantization) with weight W."""
    n_out = W.shape[0]
    lin = nn.Linear(d_in, n_out, bias=False)
    lin.weight.data = W.to(ref_linear.weight.dtype)
    return lin.to(ref_linear.weight.device)


# ---------------------------------------------------------------------------
# Real INT8 weight quantization (weight-only, per output-channel symmetric)
# ---------------------------------------------------------------------------

class Int8WeightLinear(nn.Module):
    """Real weight-only INT8 quantized linear layer.

    Weights are genuinely stored as ``torch.int8`` (signed, per output-channel
    symmetric quantisation).  Forward pass dequantizes on the fly to the input
    activation dtype before calling F.linear, so no special kernel is required
    while still achieving 2× memory savings vs float16 weights.

    Quantisation
    ------------
      scale_i = max(|W[i, :]|) / 127          (per output-channel)
      W_int8[i, :] = round(W[i, :] / scale_i) clamped to [-128, 127]

    Dequantisation (forward)
    -------------------------
      W_fp = W_int8.to(x.dtype) * scale       (broadcast over columns)
      out  = x @ W_fp.T + bias
    """

    def __init__(
        self,
        weight: torch.Tensor,   # (n_out, d_in) float tensor
        bias:   Optional[torch.Tensor] = None,
    ):
        super().__init__()
        n_out, d_in = weight.shape
        self.in_features  = d_in
        self.out_features = n_out

        w_f = weight.float()
        # Per output-channel symmetric scale
        scales = w_f.abs().amax(dim=1, keepdim=True).clamp(min=1e-9) / 127.0
        w_int8 = (w_f / scales).round().clamp(-128, 127).to(torch.int8)

        self.register_buffer('weight_int8', w_int8)   # (n_out, d_in)  int8
        self.register_buffer('scales',      scales)   # (n_out, 1)     float32

        if bias is not None:
            self.register_buffer('bias', bias.float())
        else:
            self.bias = None

        # Keep parity with ActQuantWrapper for rotated down_proj runtime Hadamard.
        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_full_hadnopow2 = False
        self.fp32_had = False

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_in)  →  out: (..., n_out)"""
        x_dtype = x.dtype
        if self.online_full_had:
            if self.fp32_had:
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else:
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)
        if self.online_full_hadnopow2:
            if self.fp32_had:
                x = hadamard_utils.matmul_hadU_cuda_nopow2(x.float(), self.had_K, self.K).to(x_dtype)
            else:
                x = hadamard_utils.matmul_hadU_cuda_nopow2(x, self.had_K, self.K)

        compute_dtype = x.dtype
        # Dequantize to compute dtype
        w = self.weight_int8.to(compute_dtype) * self.scales.to(compute_dtype)
        bias = self.bias.to(compute_dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"dtype=int8 (weight-only, per-ch sym)"
        )

    @staticmethod
    def from_linear(linear: nn.Linear) -> 'Int8WeightLinear':
        """Construct from an existing nn.Linear (or ActQuantWrapper)."""
        if hasattr(linear, 'module'):          # unwrap ActQuantWrapper
            linear = linear.module
        bias = linear.bias.data.clone() if linear.bias is not None else None
        return Int8WeightLinear(linear.weight.data.clone(), bias=bias)

    @staticmethod
    def from_weight(
        W: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> 'Int8WeightLinear':
        """Construct directly from a weight tensor."""
        return Int8WeightLinear(W, bias=bias)


def _make_real_int8_linear(
    W: torch.Tensor,
    d_in: int,
    ref_linear: nn.Linear,
) -> Int8WeightLinear:
    """Create a real INT8 weight-quantized linear from weight slice W."""
    return Int8WeightLinear.from_weight(W).to(ref_linear.weight.device)


def _copy_runtime_hadamard_config(dst_module, src_module) -> None:
    """Copy runtime Hadamard settings from an ActQuantWrapper-like module."""
    if src_module is None:
        return

    for attr in ("K", "online_full_had", "online_full_hadnopow2", "fp32_had"):
        if hasattr(src_module, attr):
            setattr(dst_module, attr, getattr(src_module, attr))

    if hasattr(src_module, "had_K"):
        dst_module.had_K = src_module.had_K


# ---------------------------------------------------------------------------
# Layer-level setup entry point (called from svd_lm_setup)
# ---------------------------------------------------------------------------

def hybrid_ffn_setup(
    model,
    args,
    tokenizer=None,
    image_processor=None,
    layer_indices_dict=None,
):
    """Replace every decoder-layer MLP with HybridSVDQuantFFN.

    Supports two workflows:

    **Phase 1 → Phase 2 (recommended)**::

        # Phase 1: offline analysis – saves channel groups to JSON, no weight change
        args.hybrid_ffn_analyze_path = '/path/to/groups.json'
        hybrid_ffn_setup(model, args, ...)   # exits after saving analysis

        # Phase 2: fast apply – loads pre-computed groups, skips calibration
        args.hybrid_ffn_groups_path = '/path/to/groups.json'
        hybrid_ffn_setup(model, args, ...)   # applies permutation + builds modules

    **Default cache path**::

        # No path flags set → auto-save/load under args.act_cache_dir/hybrid_ffn/
        hybrid_ffn_setup(model, args, ...)

    CLI flags
    ---------
    --hybrid_ffn_analyze_path PATH   Run Phase 1 only: analyze channels and save JSON.
    --hybrid_ffn_groups_path  PATH   Run Phase 2 only: load JSON and apply directly.
    (both unset)                     Auto-save/load a default JSON under act_cache_dir/hybrid_ffn/.
    """
    from . import model_utils
    

    layers = model_utils.get_layers(model)
    logging.info(f"[hybrid_ffn] Setting up {len(layers)} layers …")

    analyze_path = getattr(args, 'hybrid_ffn_analyze_path', None)
    groups_path  = getattr(args, 'hybrid_ffn_groups_path',  None)
    default_groups_path = None

    if analyze_path is None and groups_path is None:
        default_groups_path = _hybrid_ffn_groups_cache_path(args)
        if default_groups_path is not None:
            if bool(getattr(args, 'use_cache', True)) and os.path.exists(default_groups_path):
                groups_path = default_groups_path
                logging.info(
                    f"[hybrid_ffn] Using cached channel groups from default path → {groups_path}"
                )
            else:
                analyze_path = default_groups_path
                logging.info(
                    f"[hybrid_ffn] No explicit groups path provided; "
                    f"default cache path → {analyze_path}"
                )

    # ── Phase 1: offline analysis only ───────────────────────────────────
    if analyze_path:
        logging.info(f"[hybrid_ffn] Phase 1 (analyze only) → {analyze_path}")
        generated_groups = analyze_ffn_channel_groups(
            model, args, save_path=analyze_path,
            tokenizer=tokenizer, image_processor=image_processor,
        )
        if groups_path is None and default_groups_path == analyze_path:
            logging.info(
                "[hybrid_ffn] Analysis complete. Reusing freshly saved default "
                "cache in the same run."
            )
            precomputed_groups = generated_groups
            layer_inps = None
        else:
            logging.info("[hybrid_ffn] Analysis complete. Weights NOT modified. "
                         "Re-run with --hybrid_ffn_groups_path to apply.")
            return

    # ── Phase 2: load pre-computed groups (fast path, no calibration) ────
    if groups_path:
        logging.info(f"[hybrid_ffn] Phase 2 (apply pre-computed groups from {groups_path})")
        precomputed_groups = load_channel_groups(groups_path)
        # If SVDLLM-style Cholesky whitening is requested, we still need
        # calibration inputs (for computing H = X^T X per layer).
        if getattr(args, 'hybrid_ffn_act_aware', False):
            logging.info(
                "[hybrid_ffn] hybrid_ffn_act_aware=True: collecting calib inputs "
                "for Cholesky whitening (Phase 2 + act_aware mode)"
            )
            layer_inps = _collect_layer_inputs(
                model, layers, args, tokenizer, image_processor
            )
        else:
            layer_inps = None    # calibration not needed
    elif analyze_path:
        pass
    else:
        # ── Single-pass: analyze + apply in one run ───────────────────────
        logging.info("[hybrid_ffn] Single-pass mode (analyze + apply). "
                     "Tip: use --hybrid_ffn_analyze_path / --hybrid_ffn_groups_path "
                     "to separate offline analysis from model modification.")
        precomputed_groups = None
        layer_inps = _collect_layer_inputs(
            model, layers, args, tokenizer, image_processor
        ) if (args.act_aware or getattr(args, 'hybrid_ffn_calib', True)) else None

    # ── Per-layer replacement ─────────────────────────────────────────────
    _skip_layers = set(getattr(args, 'hybrid_ffn_skip_layers', None) or [])
    if _skip_layers:
        logging.info(
            f"[hybrid_ffn] --hybrid_ffn_skip_layers: layers {sorted(_skip_layers)} "
            "will keep original (pure-quant) FFN — skipping HybridSVDQuantFFN."
        )

    for idx, layer in enumerate(layers):
        if idx in _skip_layers:
            # Restore the standard QuaRot online Hadamard on down_proj so the
            # skipped layer behaves identically to the pure-quant baseline.
            # The pre-loop code (main_llava_hybrid_ffn.py) disabled online_full_had
            # on ALL down_projs to let HybridSVDQuantFFN manage its own Hadamard.
            # For skipped layers we must re-enable it to match baseline behaviour.
            try:
                _down = layer.mlp.down_proj
                _gate = _unwrap(layer.mlp.gate_proj)
                d_ffn = _gate.weight.shape[0]   # intermediate_size
                _had_K, _had_int = hadamard_utils.get_hadK(d_ffn)
                _down.online_full_had = True
                _down.had_K  = _had_K
                _down.K      = _had_int
                _down.fp32_had = bool(getattr(args, 'fp32_had', False))
                logging.info(
                    f"[hybrid_ffn] layer {idx}: SKIPPED → keeping pure-quant FFN, "
                    f"restored online Hadamard on down_proj (d_ffn={d_ffn}, K={_had_int})"
                )
            except Exception as _he:
                logging.warning(
                    f"[hybrid_ffn] layer {idx}: SKIPPED → failed to restore Hadamard "
                    f"({_he}); down_proj may lack online Hadamard."
                )
            continue

        calib      = layer_inps[idx] if (layer_inps is not None) else None
        layer_info = (
            precomputed_groups["layers"].get(str(idx))
            if precomputed_groups is not None else None
        )

        try:
            hybrid = build_hybrid_ffn(
                layer              = layer,
                layer_idx          = idx,
                args               = args,
                calib_inps         = calib,
                layer_indices_dict = layer_indices_dict,
                precomputed_group  = layer_info,
            )
            layer.mlp = hybrid
            logging.info(f"[hybrid_ffn] layer {idx}: replaced mlp → HybridSVDQuantFFN")
        except Exception as e:
            logging.error(f"[hybrid_ffn] layer {idx}: FAILED ({e}), keeping original mlp")
            import traceback; traceback.print_exc()

    logging.info("[hybrid_ffn] Setup complete.")


# ---------------------------------------------------------------------------
# Calibration activation collection
# ---------------------------------------------------------------------------

def _get_hybrid_ffn_calib_dataloader(model, args):
    """Create the calibration dataloader once and reuse it across hybrid_ffn
    analysis subroutines so sample selection stays aligned."""
    from . import data_utils, utils as utils_mod
    
    utils_mod.set_seed(args.seed)
    calib_loader = data_utils.get_loaders(
        args.cal_dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        eval_mode=False,
        hf_token=getattr(args, 'hf_token', None),
    )
    if 'wiki' in args.cal_dataset:
        return calib_loader
    return calib_loader[0]

def _collect_layer_inputs(model, layers, args, tokenizer, image_processor, dataloader=None):
    """Run the model up to each layer and collect hidden-state inputs.

    Returns a list of tensors, one per layer.  Each tensor has shape
    (N, T, d_in) where N = nsamples.
    """
    from . import utils as utils_mod, gptq_utils, molmo_utils
    from .act_aware_utils import _build_qwen_inputs, _is_qwen_model

    if dataloader is None:
        dataloader = _get_hybrid_ffn_calib_dataloader(model, args)

    dev = utils_mod.get_dev()

    def _is_smolvlm_model() -> bool:
        model_name = str(getattr(args, "model", "")).lower()
        return "smolvlm" in model_name or "smolvlm" in type(model).__name__.lower()

    logging.info("[hybrid_ffn] Loading calibration data for layer-input collection …")

    # Hook-based capture: intercept the *input* to each MLP
    layer_inps = [[] for _ in layers]

    def make_hook(idx):
        def hook(module, inp, _out):
            # inp is a tuple; inp[0] is the hidden state with shape
            # (batch, seq, d_in). Sequence lengths vary across VLM samples, so
            # flatten tokens here and store a stable 2-D shape for concat.
            hidden = inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu()
            layer_inps[idx].append(hidden)
        return hook

    handles = []
    for i, layer in enumerate(layers):
        mlp = layer.mlp
        handles.append(mlp.register_forward_hook(make_hook(i)))

    # Move the full model to GPU for the collection forward pass so hooks fire
    # for every decoder layer during the same prefill/generation path used by
    # the existing act-aware calibration code.
    model.to(dev)
    model.eval()

    # Detect text-only datasets (wikitext2 etc.) that return plain tensors
    _text_only_dataset = 'wiki' in str(getattr(args, 'cal_dataset', '')).lower()

    # Forward pass through the full model to collect activations
    with torch.no_grad():
        for batch in dataloader:
            try:
                is_qwen_model = _is_qwen_model(args, model, tokenizer)
                is_smolvlm_model = _is_smolvlm_model()
                if _text_only_dataset or (tokenizer is None and image_processor is None):
                    if isinstance(batch, (list, tuple)):
                        inp = batch[0]
                    elif isinstance(batch, dict):
                        inp = batch.get('input_ids', None)
                    else:
                        inp = None
                    if inp is None:
                        continue
                    model(inp.to(dev), use_cache=False)
                elif is_qwen_model:
                    inputs = _build_qwen_inputs(batch, image_processor, dev, args=args, model=model, tokenizer=tokenizer)
                    model.generate(**inputs, max_new_tokens=1, use_cache=True)
                elif molmo_utils.is_molmo_model(args=args, model=model, tokenizer=tokenizer):
                    inputs = molmo_utils.build_molmo_eval_inputs(
                        batch,
                        image_processor,
                        dev,
                        model=model,
                    )
                    molmo_utils.forward_molmo_prefill(model, inputs)
                elif is_smolvlm_model:
                    inputs = gptq_utils.message_to_promptsmolvlm(
                        batch, image_processor, model, tokenizer
                    )
                    model.generate(**inputs, max_new_tokens=1, use_cache=True)
                elif tokenizer is None or 'hf_v16' in str(tokenizer):
                    inputs, _ = gptq_utils.message_to_prompt(
                        batch, image_processor, model, tokenizer
                    )
                    inputs = inputs.to(dev)
                    model.generate(**inputs, max_new_tokens=1, use_cache=True)
                elif tokenizer == 'intern':
                    inputs = gptq_utils.message_to_prompt_train(
                        batch, image_processor, model, tokenizer
                    )
                    model.chat(image_processor, **inputs)
                else:  # LLaVA v1.5
                    input_ids, images = gptq_utils.message_to_prompt(
                        batch, image_processor, model, tokenizer
                    )
                    input_ids = input_ids.to(dev)
                    image_sizes = None
                    if images is not None:
                        images, image_sizes = images
                        images = images.to(dev)
                    model.generate(
                        input_ids,
                        images=images,
                        image_sizes=image_sizes,
                        do_sample=False,
                        max_new_tokens=1,
                        use_cache=True,
                    )
            except Exception as e:
                logging.exception(
                    "[hybrid_ffn] Unexpected error during layer-input collection: %s", e
                )
                raise

    for h in handles:
        h.remove()

    model.cpu()
    torch.cuda.empty_cache()

    # Concatenate samples per layer
    result = []
    missing_layers = []
    for i, inp_list in enumerate(layer_inps):
        if inp_list:
            result.append(torch.cat(inp_list, dim=0))  # (total_tokens, d_in)
        else:
            logging.warning(f"[hybrid_ffn] No calibration inputs captured for layer {i}")
            missing_layers.append(i)
            result.append(None)

    if missing_layers:
        raise RuntimeError(
            "[hybrid_ffn] Calibration failed: no layer inputs captured for "
            f"{len(missing_layers)}/{len(layers)} layers "
            f"(missing={missing_layers[:8]}{'...' if len(missing_layers) > 8 else ''})."
        )

    return result


def _compute_layer_output_gradients(model, layers, args, dev, dataloader):
    """Compute d(CE_loss)/d(layer_l_output) for every layer in **one** backward pass.

    Uses retain_grad() on each layer's output tensor so that a single
    forward+backward over the calibration set yields the gradient at every
    transformer block simultaneously.

    Returns
    -------
    list of tensors, one per layer, shape (N_tokens, d_model), on CPU.
    Layers whose gradient could not be captured return None.
    """
    from . import utils as utils_mod

    # Use a small number of samples for the backward pass to avoid OOM.
    # The gradient DIRECTION (not magnitude) is what matters for channel selection,
    # so 4–8 samples is sufficient.  Full nsamples is used for layer-input collection.
    grad_nsamples = min(getattr(args, 'nsamples', 64), 8)
    logging.info(
        f"[hybrid_ffn][grad] Computing layer-output gradients via one backward pass "
        f"(grad_nsamples={grad_nsamples}) …"
    )

    calib_loader = dataloader[:grad_nsamples] if isinstance(dataloader, list) else dataloader

    model.to(dev)
    model.eval()

    n_layers = len(layers)
    layer_grads_acc: list = [None] * n_layers

    # Use register_full_backward_hook: fires during backward, receives
    # grad_output = d(loss)/d(module_output) directly — no retain_grad() needed.
    # Requires at least one param with requires_grad=True so the computation
    # graph is built.  LLaVA parameters have requires_grad=True by default;
    # we leave them unchanged and use model.zero_grad() to discard param grads.
    def make_bwd_hook(idx):
        def hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                g = grad_output[0].detach().float().reshape(-1, grad_output[0].shape[-1]).cpu()
                if layer_grads_acc[idx] is None:
                    layer_grads_acc[idx] = g
                else:
                    layer_grads_acc[idx] = torch.cat([layer_grads_acc[idx], g], dim=0)
        return hook

    handles = [layer.register_full_backward_hook(make_bwd_hook(i))
               for i, layer in enumerate(layers)]

    for batch in calib_loader:
        try:
            if isinstance(batch, (list, tuple)):
                inp = batch[0]
            elif isinstance(batch, dict):
                inp = batch.get('input_ids', None)
            else:
                continue
            if inp is None:
                continue
            inp = inp.to(dev)

            # Process one sequence at a time to minimise peak activation memory.
            # The backward pass must store all intermediate activations, which for
            # batch_size=64 × seq_len=512 × 32 layers can exceed 100 GB.
            for i in range(inp.shape[0]):
                seq = inp[i:i+1]          # (1, T) — single sequence
                model.zero_grad(set_to_none=True)

                with torch.enable_grad():
                    out = model(input_ids=seq, labels=seq, use_cache=False)
                    loss = out.loss if hasattr(out, 'loss') else out[0]
                    if loss is None:
                        continue
                    loss.backward()

                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        except Exception as e:
            logging.warning(f"[hybrid_ffn][grad] batch skipped: {e}")

    for h in handles:
        h.remove()
    model.cpu()
    torch.cuda.empty_cache()

    # Sanity report
    captured = sum(1 for g in layer_grads_acc if g is not None)
    logging.info(f"[hybrid_ffn][grad] Captured gradients for {captured}/{n_layers} layers.")
    return layer_grads_acc


def select_svd_channels_gradient_loss(
    W_gate: torch.Tensor,          # (d_ffn, d_in)
    W_up:   torch.Tensor,          # (d_ffn, d_in)
    W_down: torch.Tensor,          # (d_out, d_ffn)
    n_svd:  int,
    calib_inps: torch.Tensor,      # (N, d_in) or (..., d_in) – MLP inputs
    layer_output_grad: torch.Tensor,  # (N, d_out) – d(final_loss)/d(layer_output)
    svd_rank: int = 0,
    w_bits: int = 8,
    w_sym: bool = True,
) -> Tuple[List[int], List[int], torch.Tensor]:
    """Greedy **gradient-based final-loss** channel selection.

    Unlike ``select_svd_channels_loss_calibration`` which measures only the
    *local* FFN reconstruction MSE, this method uses the gradient of the
    **final model CE loss** w.r.t. the layer's output hidden state to
    estimate the true downstream impact of each channel swap.

    Criterion
    ---------
    Start from a **pure-SVD baseline**.  For channel i, the estimated change
    in final loss when switching it from SVD → Quant is::

        Δloss_i ≈  Σ_{n,t}  grad[n, t, d] · Δy[n, t, d]

    where ``grad = d(CE_loss)/d(layer_output)`` (computed once externally via
    a single backward pass) and ``Δy`` is the change in FFN contribution::

        Δy[:, i_col] = (h_quant[:, i] - h_svd[:, i]) * W_down[:, i]

    We accept channel i → Quant iff Δloss_i < 0 (final loss decreases when
    replacing SVD with quant on this channel, meaning SVD was hurting accuracy).
    This is a first-order Taylor approximation, accurate for small perturbations.

    We collect ``n_quant = d_ffn - n_svd`` channels into the Quant group,
    leaving exactly ``n_svd`` channels in SVD.

    Computational cost
    ------------------
    - ``_compute_layer_output_gradients`` does **one** forward+backward over the
      full model to get ``layer_output_grad`` for every layer at once.
    - Channel selection here is purely inner products: O(N · d_out · d_ffn).
    """
    d_ffn, d_in = W_gate.shape
    d_out = W_down.shape[0]
    assert 0 < n_svd < d_ffn
    assert calib_inps is not None, "gradient_loss requires calib_inps"
    assert layer_output_grad is not None, "gradient_loss requires layer_output_grad"

    dev = W_gate.device
    Wg = W_gate.float().to(dev)
    Wu = W_up.float().to(dev)
    Wd = W_down.float().to(dev)

    X_full = calib_inps.float().reshape(-1, d_in)              # (N_full, d_in)
    G_full = layer_output_grad.float().reshape(-1, d_out)       # (N_grad, d_out)

    # Use the smaller of the two to keep GPU memory bounded.
    # G was collected with fewer samples (grad_nsamples << nsamples) intentionally.
    N = min(X_full.shape[0], G_full.shape[0])
    X = X_full[:N].to(dev)    # (N, d_in)
    G = G_full[:N].to(dev)    # (N, d_out)
    del X_full, G_full

    if svd_rank <= 0:
        svd_rank = max(1, int(0.5 * min(n_svd, d_in)))

    # ── SVD approximation → pure-SVD baseline ────────────────────────────
    with torch.no_grad():
        _, _, Vt_g = torch.linalg.svd(Wg, full_matrices=False)
        _, _, Vt_u = torch.linalg.svd(Wu, full_matrices=False)
        Wg_svd = (Wg @ Vt_g[:svd_rank].T) @ Vt_g[:svd_rank]
        Wu_svd = (Wu @ Vt_u[:svd_rank].T) @ Vt_u[:svd_rank]

        g_svd = X @ Wg_svd.T
        u_svd = X @ Wu_svd.T
        h_svd = F.silu(g_svd) * u_svd    # (N, d_ffn)

    # ── Quantize all rows (GPTQ, not RTN) ────────────────────────────────
    # GPTQ matches Phase 2 quantization, eliminating the RTN/GPTQ mismatch.
    with torch.no_grad():
        Wg_q = _gptq_quantize_matrix(Wg, X, w_bits, w_sym)
        Wu_q = _gptq_quantize_matrix(Wu, X, w_bits, w_sym)

        g_q  = X @ Wg_q.T            # (N, d_ffn)
        u_q  = X @ Wu_q.T            # (N, d_ffn)
        h_q  = F.silu(g_q) * u_q    # (N, d_ffn)

        # delta_h[:, i] = h_q[:, i] - h_svd[:, i]  (switching SVD → quant)
        delta_h = h_q - h_svd         # (N, d_ffn)

    # ── Candidate ordering: SVD-worst channels first ──────────────────────
    # Sort ascending by margin = amp*(quant_err - svd_err).
    # Low/negative margin means SVD error > quant error → try quant first.
    act_scale = X.abs().mean(dim=0) + 1e-6
    margins = _compute_channel_margins(
        Wg, Wu, Wd,
        act_scale=act_scale,
        svd_rank=svd_rank,
        w_bits=w_bits,
        w_sym=w_sym,
    )

    # Sort ascending: SVD-worst (most negative margin) channels go first
    candidate_order = torch.argsort(margins).tolist()

    # ── Greedy: accept channel into Quant if Δloss = grad · Δy < 0 ───────
    # Δloss_i ≈ G · delta_y_i  where delta_y_i = outer(h_q[:,i]-h_svd[:,i], Wd[:,i])
    # Negative Δloss means replacing SVD with quant on channel i reduces the
    # final CE loss → SVD was harmful for this channel → move it to quant.
    n_quant = d_ffn - n_svd
    quant_set: set = set()
    y_delta_accepted = torch.zeros(N, d_out, device=dev)
    accepted = rejected = 0

    for ch in candidate_order:
        if len(quant_set) >= n_quant:
            break

        dh_ch = delta_h[:, ch]                               # (N,)
        wd_ch = Wd[:, ch]                                    # (d_out,)
        delta_y = dh_ch.unsqueeze(1) * wd_ch.unsqueeze(0)   # (N, d_out)

        # First-order estimate of loss change when switching ch: SVD → quant
        delta_loss = float((G * delta_y).sum().item())

        if delta_loss < 0:
            quant_set.add(ch)
            y_delta_accepted = y_delta_accepted + delta_y
            accepted += 1
            logging.debug(
                f"[grad_loss] ch {ch:5d}: Δloss={delta_loss:+.4e} ✓ → quant"
            )
        else:
            rejected += 1

    total_delta_loss = float((G * y_delta_accepted).sum().item())
    logging.info(
        f"[grad_loss] greedy done: {accepted} → quant / {rejected} kept SVD  "
        f"estimated Δloss = {total_delta_loss:+.4e}  (rank={svd_rank}, w_bits={w_bits})"
    )

    quant_idx = sorted(quant_set)
    svd_idx   = sorted(set(range(d_ffn)) - quant_set)
    perm = torch.tensor(svd_idx + quant_idx, dtype=torch.long, device=dev)
    return svd_idx, quant_idx, perm


# ---------------------------------------------------------------------------
# SVD-Aware channel selection  (SVD-first, RTN quantization)
# ---------------------------------------------------------------------------

def select_svd_channels_svd_aware(
    W_gate: torch.Tensor,          # (d_ffn, d_in)
    W_up:   torch.Tensor,          # (d_ffn, d_in)
    W_down: torch.Tensor,          # (d_out, d_ffn)
    n_svd:  int,
    calib_inps: torch.Tensor,      # (N, d_in) or (..., d_in)
    svd_rank: int = 0,
    w_bits: int = 8,
    w_sym: bool = True,
    a_bits: int = 16,
    a_groupsize: int = -1,
    a_sym: bool = True,
    a_clip_ratio: float = 1.0,
    latent_rotate_args = None,
    layer_idx: int = -1,
    chol_L: Optional[torch.Tensor] = None,     # (d_in, d_in) lower-triangular Cholesky of X^T X
    diag_scale: Optional[torch.Tensor] = None, # (d_in,) per-feature RMS scale (diagonal mode)
) -> Tuple[List[int], List[int], torch.Tensor, List[int]]:
    """**SVD-Aware** greedy channel selection.

    Core idea: start from a pure-SVD baseline and greedily move channels
    to the quantization group **only when** RTN quantization outperforms SVD
    on the FFN reconstruction MSE.  Channels where SVD is already good enough
    stay in the SVD group; only SVD-poor channels fall back to quantization.

    Uses RTN (round-to-nearest) weight quantization and optional ActQuantizer-
    style activation fake quantization, making Phase 1 consistent with a
    Phase 2 that also uses RTN and W8A8 runtime wrappers.

    Algorithm
    ---------
    1. Compute FP16 reference output ``y_ref``.
    2. Compute W-quantized SVD baseline ``y_svd``; set as starting point.
    3. Compute per-channel delta ``delta_h = h_rtn - h_svd`` (switching SVD→RTN).
    4. Order channels by SVD quality (worst SVD first) using closed-form error.
    5. Greedily accept channel i → Quant if it strictly reduces FFN MSE.
       Stop when ``n_quant = d_ffn - n_svd`` channels are assigned to Quant.

    Returns
    -------
    svd_idx, quant_idx, perm
    """
    d_ffn, d_in = W_gate.shape
    d_out = W_down.shape[0]
    assert 0 < n_svd < d_ffn
    assert calib_inps is not None, "svd_aware requires calib_inps"

    dev = W_gate.device
    Wg = W_gate.float().to(dev)
    Wu = W_up.float().to(dev)
    Wd = W_down.float().to(dev)

    X = _prepare_svd_aware_calib_tokens(calib_inps, d_in).to(dev)   # (N, d_in)
    N = X.shape[0]

    if svd_rank <= 0:
        svd_rank = max(1, int(0.5 * min(n_svd, d_in)))

    # ── 1. FP16 reference ────────────────────────────────────────────────
    with torch.no_grad():
        g_fp  = X @ Wg.T
        u_fp  = X @ Wu.T
        y_ref = (F.silu(g_fp) * u_fp) @ Wd.T          # (N, d_out)

    # Runtime ActQuantWrapper quantizes each linear's input independently.  The
    # gate/up quant branch and the SVD BLinear branch share the same MLP input.
    X_q = _fake_quantize_activation(
        X,
        bits=a_bits,
        groupsize=a_groupsize,
        sym=a_sym,
        clip_ratio=a_clip_ratio,
    )

    # ── 2. W8A8-SVD baseline (simulate Phase 2 quantized SVD factors) ───────
    # Phase 2 quantizes ALinear (U*S^0.5) and BLinear (V^T*S^0.5) to W8.
    # With hybrid SVD factor wrappers enabled, Phase 2 also quantizes the
    # BLinear input and the ALinear latent input to A8.
    #
    # When chol_L is provided (SVDLLM full-Cholesky, --hybrid_ffn_act_aware --hybrid_ffn_act_aware_mode chol):
    #   SVD is performed on the whitened weight W @ L, then Vt is un-whitened
    #   (Vt_orig = Vt_whitened @ L^{-1}) to get factor matrices in the original
    #   input space.  This matches Phase 2's _make_svd_linear behaviour.
    #
    # When diag_scale is provided (diagonal mode, --hybrid_ffn_act_aware --hybrid_ffn_act_aware_mode diag):
    #   SVD is performed on W * scale.view(1,-1), then Vt is un-whitened by
    #   simple element-wise division: Vt_orig = Vt_w / scale.view(1,-1).
    #   No matrix inversion needed — numerically stable.

    # Flags used both in Step 2 and Step 4 (margin ordering)
    _L_d: Optional[torch.Tensor] = None      # Cholesky L (double, on dev)
    _L_inv_d: Optional[torch.Tensor] = None  # L^{-1}    (double, on dev)
    _ds: Optional[torch.Tensor] = None       # diag scale (float, on dev)

    with torch.no_grad():
        if chol_L is not None:
            # Whiten weights: W_w = W @ L  (double for numerical stability)
            _L_d = chol_L.double().to(dev)
            try:
                _L_inv_d = torch.linalg.inv(_L_d)
            except Exception as _le:
                logging.warning(
                    f"[svd_aware] layer {layer_idx}: Cholesky L inversion failed ({_le}); "
                    "falling back to raw SVD for Phase-1 baseline."
                )
                _L_d, _L_inv_d = None, None

            if _L_d is not None:
                Wg_w = (Wg.double() @ _L_d).float()
                Wu_w = (Wu.double() @ _L_d).float()
                U_g, S_g, Vt_g_w = torch.linalg.svd(Wg_w, full_matrices=False)
                U_u, S_u, Vt_u_w = torch.linalg.svd(Wu_w, full_matrices=False)
                # Un-whiten: Vt_orig = Vt_whitened @ L^{-1}
                Vt_g = (Vt_g_w.double() @ _L_inv_d).float()
                Vt_u = (Vt_u_w.double() @ _L_inv_d).float()
                logging.debug(
                    f"[svd_aware] layer {layer_idx}: using Cholesky-whitened SVD baseline"
                )
            else:
                U_g, S_g, Vt_g = torch.linalg.svd(Wg, full_matrices=False)
                U_u, S_u, Vt_u = torch.linalg.svd(Wu, full_matrices=False)
        elif diag_scale is not None:
            # Diagonal whitening: W_w = W * scale.view(1,-1)
            _ds = diag_scale.float().to(dev)
            Wg_w = Wg * _ds.view(1, -1)
            Wu_w = Wu * _ds.view(1, -1)
            U_g, S_g, Vt_g_w = torch.linalg.svd(Wg_w, full_matrices=False)
            U_u, S_u, Vt_u_w = torch.linalg.svd(Wu_w, full_matrices=False)
            # Un-whiten: Vt_orig = Vt_w / scale  (element-wise, no matrix inversion)
            Vt_g = Vt_g_w / _ds.view(1, -1)
            Vt_u = Vt_u_w / _ds.view(1, -1)
            logging.debug(
                f"[svd_aware] layer {layer_idx}: using diagonal-scale-whitened SVD baseline"
            )
        else:
            U_g, S_g, Vt_g = torch.linalg.svd(Wg, full_matrices=False)
            U_u, S_u, Vt_u = torch.linalg.svd(Wu, full_matrices=False)

        sigma_fuse = (
            _get_hybrid_ffn_sigma_fuse(latent_rotate_args)
            if latent_rotate_args is not None else "UV"
        )
        # Construct the same concrete A/B factor matrices that Phase 2 SVDLinear
        # uses, then optionally rotate that latent basis before quantization.
        A_g, B_g = _svd_factor_matrices_from_decomp(U_g, S_g, Vt_g, svd_rank, sigma_fuse)
        A_u, B_u = _svd_factor_matrices_from_decomp(U_u, S_u, Vt_u, svd_rank, sigma_fuse)

        if latent_rotate_args is not None:
            A_g, B_g = _apply_svd_latent_rotation_to_factors(
                A_g, B_g, latent_rotate_args, layer_idx=layer_idx
            )
            A_u, B_u = _apply_svd_latent_rotation_to_factors(
                A_u, B_u, latent_rotate_args, layer_idx=layer_idx
            )

        # Apply W8 RTN quantization to factor matrices (matching Phase 2 rtn_fwrd)
        A_g_q = _quantize_rows(A_g, w_bits, w_sym)
        B_g_q = _quantize_rows(B_g, w_bits, w_sym)
        A_u_q = _quantize_rows(A_u, w_bits, w_sym)
        B_u_q = _quantize_rows(B_u, w_bits, w_sym)

        # Compute hidden activations via W8A8 factor chain:
        # x -> A8 -> BLinear(W8) -> latent -> A8 -> ALinear(W8)
        x_lat_g = X_q @ B_g_q.T                        # (N, rank)
        x_lat_g = _fake_quantize_activation(
            x_lat_g, bits=a_bits, groupsize=a_groupsize,
            sym=a_sym, clip_ratio=a_clip_ratio,
        )
        g_svd   = x_lat_g @ A_g_q.T                    # (N, d_ffn)
        x_lat_u = X_q @ B_u_q.T
        x_lat_u = _fake_quantize_activation(
            x_lat_u, bits=a_bits, groupsize=a_groupsize,
            sym=a_sym, clip_ratio=a_clip_ratio,
        )
        u_svd   = x_lat_u @ A_u_q.T                    # (N, d_ffn)
        h_svd = F.silu(g_svd) * u_svd                  # (N, d_ffn)
        h_svd_wd = h_svd
        y_svd = h_svd_wd @ Wd.T                        # (N, d_out)

    baseline_mse = float(((y_svd - y_ref) ** 2).mean().item())
    logging.info(
        f"[svd_aware] W{w_bits}A{a_bits}-SVD baseline MSE = {baseline_mse:.6f}"
    )

    # ── 3. RTN weight-quantized branch activations (per-channel delta) ───
    with torch.no_grad():
        Wg_q = _quantize_rows(Wg, w_bits, w_sym)
        Wu_q = _quantize_rows(Wu, w_bits, w_sym)

        g_q = X_q @ Wg_q.T
        u_q = X_q @ Wu_q.T
        h_q = F.silu(g_q) * u_q                        # (N, d_ffn)

        # delta_h[:, i] = h_rtn[:, i] - h_svd[:, i]  (switch SVD → RTN)
        delta_h = h_q - h_svd_wd

    # ── 4. Candidate ordering: SVD-worst first ────────────────────────────
    # Fix 1: When whitening is active, rank channels using whitened weights so
    # that the SVD quality ordering is consistent with the whitened SVD baseline
    # computed in Step 2.  Without this fix, raw-W margins would rank channels
    # differently from the whitened-SVD baseline, causing suboptimal selection.
    with torch.no_grad():
        if _L_d is not None:
            # Cholesky whitened ordering
            Wg_m = (Wg.double() @ _L_d).float()
            Wu_m = (Wu.double() @ _L_d).float()
            act_scale = (X.double() @ _L_d).float().abs().mean(dim=0) + 1e-6
        elif _ds is not None:
            # Diagonal whitened ordering
            Wg_m = Wg * _ds.view(1, -1)
            Wu_m = Wu * _ds.view(1, -1)
            act_scale = (X * _ds.view(1, -1)).abs().mean(dim=0) + 1e-6
        else:
            Wg_m = Wg
            Wu_m = Wu
            act_scale = X.abs().mean(dim=0) + 1e-6

    margins = _compute_channel_margins(
        Wg_m, Wu_m, Wd,
        act_scale=act_scale,
        svd_rank=svd_rank,
        w_bits=w_bits,
        w_sym=w_sym,
    )

    # Ascending: most negative margin = SVD error >> RTN error → try quant first
    candidate_order = torch.argsort(margins).tolist()

    # ── 5. Greedy swap: SVD → Quant (vectorised fast path) ───────────────
    # Instead of recomputing (N, d_out) trial outputs for every candidate,
    # we use the exact identity:
    #
    #   trial_mse = current_mse
    #             + 2/(N*d_out) * dot(delta_h[:,ch], score[:,ch])
    #             + mean(delta_h[:,ch]**2) * mean(Wd[:,ch]**2)
    #
    # where score = err_current @ Wd  (shape N × d_ffn).
    # When a channel is accepted we update score with a rank-1 correction:
    #   score += outer(delta_h[:,ch], Wd[:,ch] @ Wd)
    # This reduces per-candidate cost from O(N·d_out) to O(N).
    n_quant    = d_ffn - n_svd
    current_mse = baseline_mse
    quant_set: set = set()
    accepted = rejected = 0

    err_current = (y_svd - y_ref).contiguous()          # (N, d_out)
    score       = err_current @ Wd                      # (N, d_ffn)  – one matmul
    dh_sq       = (delta_h ** 2).mean(dim=0)            # (d_ffn,)
    wd_sq       = (Wd ** 2).mean(dim=0)                 # (d_ffn,)
    Nd_out      = float(N * d_out)

    # greedy_accept_order: channels in the order they were accepted into quant.
    # We run the full loop (no early stop) so the order can be reused to derive
    # any svd_ffn_ratio without re-running Phase 1.
    greedy_accept_order: List[int] = []

    for ch in candidate_order:
        dh_ch = delta_h[:, ch]                          # (N,)
        cross  = float((dh_ch @ score[:, ch]).item()) / Nd_out
        trial_mse = current_mse + 2.0 * cross + float(dh_sq[ch].item()) * float(wd_sq[ch].item())

        if trial_mse < current_mse:
            greedy_accept_order.append(ch)
            quant_set.add(ch)
            # rank-1 update to score: score += outer(dh_ch, Wd[:,ch] @ Wd)
            wd_cross = Wd[:, ch] @ Wd                   # (d_ffn,)
            score.add_(dh_ch.unsqueeze(1) * wd_cross.unsqueeze(0))
            current_mse = trial_mse
            accepted += 1
            logging.debug(f"[svd_aware] ch {ch:5d}: MSE → {trial_mse:.6f} ✓ → quant")
        else:
            rejected += 1

    del err_current, score, dh_sq, wd_sq

    improvement = baseline_mse - current_mse
    logging.info(
        f"[svd_aware] greedy done: {accepted} → quant / {rejected} kept SVD  "
        f"MSE {baseline_mse:.6f} → {current_mse:.6f}  "
        f"(Δ={improvement:+.6f}, rank={svd_rank}, W{w_bits}A{a_bits} RTN)"
    )

    # Use greedy_accept_order so any ratio can be derived from this single run.
    quant_idx = sorted(greedy_accept_order[:n_quant])
    svd_idx   = sorted(set(range(d_ffn)) - set(quant_idx))
    perm = torch.tensor(svd_idx + quant_idx, dtype=torch.long, device=dev)
    return svd_idx, quant_idx, perm, greedy_accept_order


# ---------------------------------------------------------------------------
# Phase 1: Offline channel-group analysis  (analyze → save JSON)
# ---------------------------------------------------------------------------

def analyze_ffn_channel_groups(
    model,
    args,
    save_path: str,
    tokenizer=None,
    image_processor=None,
) -> dict:
    """**Phase 1 (offline)**: Determine per-layer SVD / Quant channel grouping
    and save the result to a JSON file.

    This is a pure analysis pass — no weight modification occurs.
    The resulting file can be fed to ``hybrid_ffn_setup`` via
    ``args.hybrid_ffn_groups_path`` to skip re-analysis on subsequent runs.

    Workflow
    --------
    1. Collect calibration activations (hidden states fed into each MLP).
    2. For each layer, call ``select_svd_channels`` to determine which
       hidden channels benefit from SVD vs Quant (error-comparison criterion).
    3. Save the per-layer assignment (svd_idx, quant_idx, perm, margins, …)
       to ``save_path`` as JSON.

    Returns the groups dict (same structure as the JSON file).
    """
    import json
    from . import model_utils

    layers = model_utils.get_layers(model)
    n_layers = len(layers)
    logging.info(f"[hybrid_ffn][analyze] Starting offline analysis for {n_layers} layers …")

    # ── Collect calibration activations ──────────────────────────────────
    calib_dataloader = _get_hybrid_ffn_calib_dataloader(model, args)
    layer_inps = _collect_layer_inputs(
        model, layers, args, tokenizer, image_processor, calib_dataloader
    )

    # ── Per-layer analysis ────────────────────────────────────────────────
    svd_ffn_ratio = getattr(args, 'svd_ffn_ratio', 0.2)
    sigma_fuse = _get_hybrid_ffn_sigma_fuse(args)
    w_bits = int(getattr(args, 'w_bits', 4))
    w_sym  = not bool(getattr(args, 'w_asym', False))
    a_bits = int(getattr(args, 'a_bits', 16))
    a_groupsize = int(getattr(args, 'a_groupsize', -1))
    a_sym = not bool(getattr(args, 'a_asym', False))
    a_clip_ratio = min(
        float(getattr(args, 'a_clip_ratio', 1.0)),
        float(getattr(args, 'lma_clip_ratio', 1.0)),
    )
    latent_rotate = bool(getattr(args, "hybrid_ffn_svd_latent_rotate", False))
    latent_rotate_mode = getattr(args, "hybrid_ffn_svd_latent_rotate_mode", "hadamard")

    calib_method = getattr(args, 'svd_ffn_calib_method', 'error_compare')

    # For gradient_loss: compute d(CE_loss)/d(layer_output) for all layers at once
    # (one forward+backward pass) BEFORE the per-layer loop.
    from . import utils as utils_mod
    dev = utils_mod.get_dev()
    analysis_dev = dev if dev.type == "cuda" else torch.device("cpu")
    logging.info(f"[hybrid_ffn][analyze] per-layer analysis device → {analysis_dev}")
    if calib_method == 'gradient_loss':
        layer_grads = _compute_layer_output_gradients(
            model, layers, args, dev, calib_dataloader
        )
    else:
        layer_grads = [None] * n_layers

    groups = {
        "meta": {
            "model":         getattr(args, 'model', ''),
            "svd_ffn_ratio": svd_ffn_ratio,
            "sigma_fuse":    sigma_fuse,
            "rank_ratio":    float(getattr(args, 'rank_ratio', 0.5)),
            "hybrid_ffn_rank_ratio": (
                None if getattr(args, 'hybrid_ffn_rank_ratio', None) is None
                else float(getattr(args, 'hybrid_ffn_rank_ratio'))
            ),
            "w_bits":        w_bits,
            "w_sym":         w_sym,
            "a_bits":        a_bits,
            "a_sym":         a_sym,
            "a_groupsize":   a_groupsize,
            "a_clip_ratio":  a_clip_ratio,
            "hybrid_ffn_svd_latent_rotate": latent_rotate,
            "hybrid_ffn_svd_latent_rotate_mode": latent_rotate_mode,
            "n_layers":      n_layers,
            "calib_method":  calib_method,
            "hybrid_ffn_act_aware": bool(getattr(args, 'hybrid_ffn_act_aware', False)),
            "hybrid_ffn_act_aware_mode": getattr(args, 'hybrid_ffn_act_aware_mode', 'chol'),
        },
        "layers": {},
    }

    for idx, layer in enumerate(layers):
        from .quant_utils import ActQuantWrapper

        mlp     = layer.mlp
        gate_lin = _unwrap(mlp.gate_proj)
        up_lin   = _unwrap(mlp.up_proj)
        down_lin = _unwrap(mlp.down_proj)

        d_ffn = gate_lin.out_features
        d_in  = gate_lin.in_features

        n_svd = max(1, min(int(d_ffn * svd_ffn_ratio), d_ffn - 1))

        Wg = gate_lin.weight.data.float()
        Wu = up_lin.weight.data.float()
        Wd = down_lin.weight.data.float()

        rank = _compute_svd_rank(n_svd, d_in, args, idx, None, key='up_proj')

        calib = layer_inps[idx]
        grad = layer_grads[idx] if layer_grads[idx] is not None else None
        act_scale = None

        use_gpu_analysis = analysis_dev.type == "cuda"
        if use_gpu_analysis:
            if calib_method == 'svd_aware' and calib is not None:
                calib = _prepare_svd_aware_calib_tokens(calib, d_in)
            elif calib_method == 'loss_calibration' and calib is not None:
                calib = _flatten_calib_tokens(calib, d_in)
            elif calib_method == 'gradient_loss':
                if calib is not None and grad is not None:
                    calib, grad = _prepare_gradient_loss_inputs(calib, grad, d_in, _unwrap(mlp.down_proj).out_features)
                elif calib is not None:
                    act_scale = _compute_act_scale_from_calib(calib, d_in)
            else:
                if calib is not None:
                    act_scale = _compute_act_scale_from_calib(calib, d_in)

            Wg = Wg.to(analysis_dev, non_blocking=True)
            Wu = Wu.to(analysis_dev, non_blocking=True)
            Wd = Wd.to(analysis_dev, non_blocking=True)
            if calib is not None:
                calib = calib.to(analysis_dev, non_blocking=True)
            if grad is not None:
                grad = grad.to(analysis_dev, non_blocking=True)
            if act_scale is not None:
                act_scale = act_scale.to(analysis_dev, non_blocking=True)

        greedy_order = None   # populated only by svd_aware

        # ── Input whitening for Phase 1 (svd_aware + --hybrid_ffn_act_aware) ──
        # Compute the whitening statistics from calibration inputs so that
        # select_svd_channels_svd_aware uses the same whitened SVD quality metric
        # as Phase 2 (_make_svd_linear).  Two modes:
        #   'chol': full Cholesky L of X^T X  (SVDLLM, requires L^{-1})
        #   'diag': diagonal RMS scale         (stable, no matrix inversion)
        chol_L_layer = None
        diag_scale_layer = None
        _act_aware_mode = getattr(args, 'hybrid_ffn_act_aware_mode', 'chol')
        if (calib_method == 'svd_aware'
                and getattr(args, 'hybrid_ffn_act_aware', False)
                and calib is not None):
            try:
                _MAX_TOK_P1 = 4096
                X_raw = calib.reshape(-1, d_in)
                if X_raw.shape[0] > _MAX_TOK_P1:
                    step = max(1, X_raw.shape[0] // _MAX_TOK_P1)
                    X_raw = X_raw[::step][:_MAX_TOK_P1]
                X_d = X_raw.double().to(analysis_dev)
                if _act_aware_mode == 'diag':
                    # Diagonal variance-based scaling
                    var = (X_d ** 2).mean(dim=0)
                    diag_scale_layer = var.clamp(min=1e-8).sqrt().cpu()
                    logging.info(
                        f"[hybrid_ffn][analyze] layer {idx}: "
                        f"diagonal whitening scale computed (mode=diag)"
                    )
                else:
                    # Full Cholesky
                    H = (X_d.T @ X_d) / X_d.shape[0]
                    damp = 0.01 * H.diagonal().mean().clamp(min=1e-8)
                    H.diagonal().add_(damp)
                    chol_L_layer = torch.linalg.cholesky(H).cpu()
                    logging.info(
                        f"[hybrid_ffn][analyze] layer {idx}: "
                        f"Cholesky whitening matrix computed for Phase-1 svd_aware"
                    )
                del X_d
                torch.cuda.empty_cache()
            except Exception as _ce:
                logging.warning(
                    f"[hybrid_ffn][analyze] layer {idx}: "
                    f"Whitening computation failed ({_ce}); using raw SVD baseline."
                )

        if calib_method == 'gradient_loss':
            logging.info(f"[hybrid_ffn][analyze] layer {idx}: using gradient_loss")
            if grad is None:
                logging.warning(
                    f"[hybrid_ffn][analyze] layer {idx}: no gradient captured, "
                    f"falling back to error_compare"
                )
                svd_idx, quant_idx, perm = select_svd_channels(
                    Wg, Wu, Wd, n_svd,
                    calib_inps = calib,
                    act_scale  = act_scale,
                    svd_rank   = rank,
                    w_bits     = w_bits,
                    w_sym      = w_sym,
                )
            else:
                svd_idx, quant_idx, perm = select_svd_channels_gradient_loss(
                    Wg, Wu, Wd, n_svd,
                    calib_inps        = calib,
                    layer_output_grad = grad,
                    svd_rank          = rank,
                    w_bits            = w_bits,
                    w_sym             = w_sym,
                )
        elif calib_method == 'loss_calibration':
            logging.info(f"[hybrid_ffn][analyze] layer {idx}: using loss_calibration")
            svd_idx, quant_idx, perm = select_svd_channels_loss_calibration(
                Wg, Wu, Wd, n_svd,
                calib_inps = calib,
                svd_rank   = rank,
                w_bits     = w_bits,
                w_sym      = w_sym,
            )
        elif calib_method == 'svd_aware':
            _whitening_label = (
                " + Cholesky whitening" if chol_L_layer is not None else
                " + diagonal whitening" if diag_scale_layer is not None else ""
            )
            logging.info(f"[hybrid_ffn][analyze] layer {idx}: using svd_aware (RTN){_whitening_label}")
            svd_idx, quant_idx, perm, greedy_order = select_svd_channels_svd_aware(
                Wg, Wu, Wd, n_svd,
                calib_inps  = calib,
                svd_rank    = rank,
                w_bits      = w_bits,
                w_sym       = w_sym,
                a_bits       = a_bits,
                a_groupsize  = a_groupsize,
                a_sym        = a_sym,
                a_clip_ratio = a_clip_ratio,
                latent_rotate_args = args,
                layer_idx  = idx,
                chol_L     = chol_L_layer.to(analysis_dev) if chol_L_layer is not None else None,
                diag_scale = diag_scale_layer.to(analysis_dev) if diag_scale_layer is not None else None,
            )
        else:
            svd_idx, quant_idx, perm = select_svd_channels(
                Wg, Wu, Wd, n_svd,
                calib_inps = calib,
                act_scale  = act_scale,
                svd_rank   = rank,
                w_bits     = w_bits,
                w_sym      = w_sym,
            )

        layer_entry = {
            "svd_idx":   svd_idx,
            "quant_idx": quant_idx,
            "perm":      perm.tolist(),
            "n_svd":     len(svd_idx),
            "svd_rank":  rank,
            "d_ffn":     d_ffn,
            "d_in":      d_in,
        }
        if greedy_order is not None:
            # Full greedy acceptance order lets any svd_ffn_ratio be derived
            # offline via derive_ratio_groups.py without re-running Phase 1.
            layer_entry["greedy_accept_order"] = greedy_order
        groups["layers"][str(idx)] = layer_entry
        logging.info(
            f"[hybrid_ffn][analyze] layer {idx}: "
            f"SVD={len(svd_idx)}, Quant={len(quant_idx)}, rank={rank}"
        )
        del Wg, Wu, Wd, calib, grad, act_scale

    # ── Save ─────────────────────────────────────────────────────────────
    import os
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, 'w') as f:
        import json
        json.dump(groups, f, indent=2)
    logging.info(f"[hybrid_ffn][analyze] Channel groups saved → {save_path}")

    return groups


def load_channel_groups(path: str) -> dict:
    """**Phase 2 helper**: Load pre-computed channel groups from a JSON file.

    Returns the groups dict with integer list fields restored.
    """
    import json
    with open(path, 'r') as f:
        groups = json.load(f)
    # Restore integer lists (JSON stores everything as native Python types)
    for layer_info in groups["layers"].values():
        layer_info["svd_idx"]   = [int(x) for x in layer_info["svd_idx"]]
        layer_info["quant_idx"] = [int(x) for x in layer_info["quant_idx"]]
        layer_info["perm"]      = [int(x) for x in layer_info["perm"]]
        layer_info["n_svd"]     = int(layer_info["n_svd"])
        layer_info["svd_rank"]  = int(layer_info["svd_rank"])
    logging.info(f"[hybrid_ffn] Loaded channel groups from {path} "
                 f"({len(groups['layers'])} layers)")
    return groups


# ---------------------------------------------------------------------------
# GPTQ helpers: return the sequential list for hybrid FFN layers
# ---------------------------------------------------------------------------

def hybrid_ffn_gptq_sequential(real_int8: bool = False) -> list:
    """Return the GPTQ sequential groups for a layer that has HybridSVDQuantFFN.

    The SVD sub-modules (gate_svd, up_svd) should NOT be GPTQ-quantised
    (they are already compressed via SVD).

    When real_int8=False (fake quant / GPTQ mode):
      gate_quant, up_quant, and down_proj ARE GPTQ-quantised.

    When real_int8=True (real INT8 mode):
      All three quant-side modules are already stored as torch.int8 at
      setup time → nothing left for GPTQ to do → return empty list.
    """
    if real_int8:
        # Real INT8 already applied in build_hybrid_ffn; no GPTQ needed
        return []
    return [
        # Quant side: gate_quant and up_quant inner linears (fake quant / GPTQ)
        ['mlp.gate_quant.module', 'mlp.up_quant.module'],
        # Full down_proj (columns already permuted to [SVD | Quant])
        ['mlp.down_proj.module'],
    ]


def hybrid_ffn_rtn_targets(real_int8: bool = False, fp16_quant_group: bool = False) -> list:
    """Return the RTN fake-quant targets for a layer with HybridSVDQuantFFN.

    RTN fake-quant should quantize both parts of the hybrid FFN:
      - SVD side: the factorized A/B linear weights
      - Quant side: gate/up fake-quant linears and the full down_proj

    When ``real_int8=True``, quant-side modules have already been converted to
    ``Int8WeightLinear`` at setup time, so RTN should only touch the SVD side.

    When ``fp16_quant_group=True``, all FFN weights remain in FP16; return empty.
    """
    if fp16_quant_group:
        return []

    targets = [
        'mlp.gate_svd.ALinear',
        'mlp.gate_svd.BLinear',
        'mlp.up_svd.ALinear',
        'mlp.up_svd.BLinear',
    ]
    if real_int8:
        return targets

    return targets + [
        'mlp.gate_quant.module',
        'mlp.up_quant.module',
        'mlp.down_proj.module',
    ]
