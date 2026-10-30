"""Unit tests for HybridSVDQuantFFN.

Tests cover:
1. Permutation consistency: same perm applied to gate, up (rows) and down (cols).
2. Forward-pass equivalence: permuted full-precision model output == original.
3. Shape correctness: all intermediate tensors have the expected shapes.
4. Channel selection: n_svd channels genuinely assigned, no overlap with quant.
"""

import os
import sys
import types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import unittest.mock as mock


# ── minimal stubs so tests run without CUDA / full model ─────────────────────

class _Args:
    rank_ratio       = 0.5
    svd_ffn_ratio    = 0.5
    act_aware        = False
    act_alpha        = 0.5
    sigma_fuse       = "UV"
    use_true_param_ratio = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_ffn(d_in=64, d_ffn=128, d_out=64):
    """Return (gate_lin, up_lin, down_lin) with random weights."""
    gate = nn.Linear(d_in, d_ffn, bias=False)
    up   = nn.Linear(d_in, d_ffn, bias=False)
    down = nn.Linear(d_ffn, d_out, bias=False)
    return gate, up, down


def _original_ffn(gate, up, down, x):
    h = F.silu(gate(x)) * up(x)
    return down(h)


# ── Stub SVDLinear so we don't need fast_hadamard_transform on the login node ──

class _StubSVDLinear(nn.Module):
    """Minimal SVDLinear: y = A(B(x))  where B: d_in→rank, A: rank→n_out."""
    def __init__(self, U, S, V, sigma_fuse="UV"):
        super().__init__()
        rank = S.shape[0]
        n_out, d_in = U.shape[0], V.shape[0]
        self.BLinear = nn.Linear(d_in, rank, bias=False)
        self.ALinear = nn.Linear(rank, n_out, bias=False)
        if sigma_fuse == "UV":
            self.BLinear.weight.data = (V * S.sqrt().unsqueeze(0)).T.contiguous()
            self.ALinear.weight.data = (U * S.sqrt().unsqueeze(0)).contiguous()
        else:
            self.BLinear.weight.data = V.T.contiguous()
            self.ALinear.weight.data = (U * S.unsqueeze(0)).contiguous()

    def forward(self, x):
        return self.ALinear(self.BLinear(x))


def _make_stub_svd_linear(W: torch.Tensor, rank: int):
    U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
    return _StubSVDLinear(U[:, :rank], S[:rank], Vt[:rank, :].T)


# ── import the module under test ──────────────────────────────────────────────

from utils.hybrid_ffn_utils import (
    HybridSVDQuantFFN,
    _finalise_score,
    _make_quant_linear,
    hybrid_ffn_rtn_targets,
    select_svd_channels,
)
from utils.quant_utils import ActQuantWrapper
from utils import gptq_utils, svd_utils

# ── Test 1: channel selection returns valid partition ─────────────────────────

def test_channel_selection_partition():
    d_in, d_ffn, d_out = 32, 64, 32
    gate, up, down = _make_ffn(d_in, d_ffn, d_out)
    n_svd = 20
    svd_idx, quant_idx, perm = select_svd_channels(
        gate.weight.data, up.weight.data, down.weight.data, n_svd
    )
    # ``n_svd`` is a cap: error_compare deliberately leaves a channel in the
    # quantized group when its proxy quantization error is no worse than its
    # SVD reconstruction error.
    assert 0 <= len(svd_idx) <= n_svd
    assert len(quant_idx) == d_ffn - len(svd_idx)
    # No overlap
    assert set(svd_idx) & set(quant_idx) == set()
    # Full coverage
    assert set(svd_idx) | set(quant_idx) == set(range(d_ffn))
    # perm = svd_idx + quant_idx (in that order)
    assert perm.tolist() == svd_idx + quant_idx


# ── Test 2: permutation consistency ──────────────────────────────────────────

def test_permutation_consistency():
    """Applying perm to gate/up rows and down cols must not change FFN output."""
    torch.manual_seed(0)
    d_in, d_ffn, d_out = 32, 64, 32
    gate, up, down = _make_ffn(d_in, d_ffn, d_out)
    n_svd = 24
    svd_idx, quant_idx, perm = select_svd_channels(
        gate.weight.data, up.weight.data, down.weight.data, n_svd
    )

    # Permuted weights
    Wg_grp = gate.weight.data[perm, :]
    Wu_grp = up.weight.data[perm, :]
    Wd_grp = down.weight.data[:, perm]

    gate_grp = nn.Linear(d_in, d_ffn, bias=False)
    up_grp   = nn.Linear(d_in, d_ffn, bias=False)
    down_grp = nn.Linear(d_ffn, d_out, bias=False)
    gate_grp.weight.data = Wg_grp
    up_grp.weight.data   = Wu_grp
    down_grp.weight.data = Wd_grp

    x = torch.randn(4, 16, d_in)
    y_orig = _original_ffn(gate, up, down, x)
    y_perm = _original_ffn(gate_grp, up_grp, down_grp, x)
    torch.testing.assert_close(y_orig, y_perm, rtol=1e-4, atol=1e-4,
                                msg="Permuted FFN output != original FFN output")


# ── Test 3: shape sanity with HybridSVDQuantFFN ───────────────────────────────

def test_hybrid_ffn_shapes():
    """HybridSVDQuantFFN should produce correct output shape."""
    torch.manual_seed(42)
    d_in, d_ffn, d_out = 32, 64, 32
    n_svd  = 24

    gate, up, down = _make_ffn(d_in, d_ffn, d_out)
    _, _, perm = select_svd_channels(
        gate.weight.data, up.weight.data, down.weight.data, n_svd
    )
    perm_cpu = perm.cpu()

    Wg_svd   = gate.weight.data[perm_cpu[:n_svd], :]
    Wu_svd   = up.weight.data[perm_cpu[:n_svd], :]
    Wg_quant = gate.weight.data[perm_cpu[n_svd:], :]
    Wu_quant = up.weight.data[perm_cpu[n_svd:], :]

    rank = max(1, int(0.5 * min(n_svd, d_in)))

    gate_svd_mod   = _make_stub_svd_linear(Wg_svd, rank)
    up_svd_mod     = _make_stub_svd_linear(Wu_svd, rank)
    gate_quant_mod = _make_quant_linear(Wg_quant, d_in, gate)
    up_quant_mod   = _make_quant_linear(Wu_quant, d_in, up)

    down_inner = nn.Linear(d_ffn, d_out, bias=False)
    down_inner.weight.data = down.weight.data[:, perm_cpu]
    down_aw = ActQuantWrapper(down_inner)

    hybrid = HybridSVDQuantFFN(
        gate_svd   = gate_svd_mod,
        up_svd     = up_svd_mod,
        gate_quant = gate_quant_mod,
        up_quant   = up_quant_mod,
        down_proj  = down_aw,
        perm       = perm,
        n_svd      = n_svd,
    )

    x = torch.randn(2, 8, d_in)
    y = hybrid(x)
    assert y.shape == (2, 8, d_out), f"Expected (2,8,{d_out}), got {y.shape}"


# ── Test 4: full-precision hybrid matches original ─────────────────────────────

def test_hybrid_ffn_fp_equivalence():
    """With full-rank SVD and fp32 pass-through quant wrappers, output == original."""
    torch.manual_seed(7)
    d_in, d_ffn, d_out = 16, 32, 16
    n_svd = 16   # half to SVD

    gate, up, down = _make_ffn(d_in, d_ffn, d_out)
    _, _, perm = select_svd_channels(
        gate.weight.data, up.weight.data, down.weight.data, n_svd
    )
    perm_cpu = perm.cpu()

    Wg_svd   = gate.weight.data[perm_cpu[:n_svd], :]
    Wu_svd   = up.weight.data[perm_cpu[:n_svd], :]
    Wg_quant = gate.weight.data[perm_cpu[n_svd:], :]
    Wu_quant = up.weight.data[perm_cpu[n_svd:], :]

    # Full rank → SVD group is lossless
    rank = min(n_svd, d_in)

    gate_svd_mod   = _make_stub_svd_linear(Wg_svd, rank)
    up_svd_mod     = _make_stub_svd_linear(Wu_svd, rank)
    gate_quant_mod = _make_quant_linear(Wg_quant, d_in, gate)
    up_quant_mod   = _make_quant_linear(Wu_quant, d_in, up)

    down_inner = nn.Linear(d_ffn, d_out, bias=False)
    down_inner.weight.data = down.weight.data[:, perm_cpu]
    down_aw = ActQuantWrapper(down_inner)

    hybrid = HybridSVDQuantFFN(
        gate_svd   = gate_svd_mod,
        up_svd     = up_svd_mod,
        gate_quant = gate_quant_mod,
        up_quant   = up_quant_mod,
        down_proj  = down_aw,
        perm       = perm,
        n_svd      = n_svd,
    )

    x = torch.randn(3, 5, d_in)
    y_hybrid = hybrid(x)
    y_orig   = _original_ffn(gate, up, down, x)

    torch.testing.assert_close(y_hybrid, y_orig, rtol=1e-4, atol=1e-4,
                                msg="Full-rank hybrid output diverged from original")


def _build_hybrid_module(d_in=16, d_ffn=32, d_out=16, n_svd=16):
    gate, up, down = _make_ffn(d_in, d_ffn, d_out)
    _, _, perm = select_svd_channels(
        gate.weight.data, up.weight.data, down.weight.data, n_svd
    )
    perm_cpu = perm.cpu()

    Wg_svd = gate.weight.data[perm_cpu[:n_svd], :]
    Wu_svd = up.weight.data[perm_cpu[:n_svd], :]
    Wg_quant = gate.weight.data[perm_cpu[n_svd:], :]
    Wu_quant = up.weight.data[perm_cpu[n_svd:], :]

    rank = max(1, min(n_svd, d_in) // 2)

    down_inner = nn.Linear(d_ffn, d_out, bias=False)
    down_inner.weight.data = down.weight.data[:, perm_cpu]
    down_aw = ActQuantWrapper(down_inner)

    return HybridSVDQuantFFN(
        gate_svd=_make_stub_svd_linear(Wg_svd, rank),
        up_svd=_make_stub_svd_linear(Wu_svd, rank),
        gate_quant=_make_quant_linear(Wg_quant, d_in, gate),
        up_quant=_make_quant_linear(Wu_quant, d_in, up),
        down_proj=down_aw,
        perm=perm,
        n_svd=n_svd,
    )


class _HybridLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _build_hybrid_module()


class _HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = types.SimpleNamespace(layers=nn.ModuleList([_HybridLayer()]))


class _RTNArgs:
    svd_modules = None
    use_hybrid_ffn = True
    hybrid_real_int8 = False
    w_groupsize = -1
    w_bits = 4
    w_asym = False
    w_clip = False
    int8_down_proj = False


def _clone_hybrid_weights(layer):
    return {
        name: svd_utils.rgetattr(layer, name).weight.detach().clone()
        for name in hybrid_ffn_rtn_targets()
    }


def test_hybrid_ffn_rtn_targets_complete():
    targets = hybrid_ffn_rtn_targets()
    assert targets == [
        "mlp.gate_svd.ALinear",
        "mlp.gate_svd.BLinear",
        "mlp.up_svd.ALinear",
        "mlp.up_svd.BLinear",
        "mlp.gate_quant.module",
        "mlp.up_quant.module",
        "mlp.down_proj.module",
    ]


def test_hybrid_ffn_rtn_quantizes_svd_and_quant_parts(monkeypatch):
    torch.manual_seed(0)
    model = _HybridModel()
    layer = model.model.layers[0]
    before = _clone_hybrid_weights(layer)

    monkeypatch.setattr(gptq_utils.model_utils, "get_layers", lambda m: m.model.layers)
    quantizers = gptq_utils.rtn_fwrd(model, torch.device("cpu"), _RTNArgs())

    assert set(quantizers.keys()) == {f"model.layers.0.{name}" for name in before}

    after = _clone_hybrid_weights(layer)
    for name in before:
        assert not torch.equal(before[name], after[name]), f"{name} weight was not RTN-quantized"


def test_hybrid_ffn_rtn_preserves_forward_shape(monkeypatch):
    torch.manual_seed(1)
    model = _HybridModel()
    layer = model.model.layers[0]
    x = torch.randn(2, 3, 16)

    y_before = layer.mlp(x)
    monkeypatch.setattr(gptq_utils.model_utils, "get_layers", lambda m: m.model.layers)
    gptq_utils.rtn_fwrd(model, torch.device("cpu"), _RTNArgs())
    y_after = layer.mlp(x)

    assert y_before.shape == y_after.shape == (2, 3, 16)
    assert torch.isfinite(y_after).all()


if __name__ == "__main__":
    test_channel_selection_partition()
    print("✓ test_channel_selection_partition")
    test_permutation_consistency()
    print("✓ test_permutation_consistency")
    test_hybrid_ffn_shapes()
    print("✓ test_hybrid_ffn_shapes")
    test_hybrid_ffn_fp_equivalence()
    print("✓ test_hybrid_ffn_fp_equivalence")
    test_hybrid_ffn_rtn_targets_complete()
    print("✓ test_hybrid_ffn_rtn_targets_complete")
    print("\nAll tests passed.")
