import sys
from pathlib import Path
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

KERNELS_DIR = Path(__file__).resolve().parents[2]
FD_KERNEL_DIR = KERNELS_DIR / "layer_bench" / "fd_kernel"
if str(FD_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(FD_KERNEL_DIR))

from destindex_copy_quantize_latent import destindex_copy_quantize_latent
from flash_decoding import token_decode_attention_flash_int8_latent_per_head_decoding_rope

from .utils import _rms_norm


def _cached_sdpa(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
    # The queries are the last q_len positions of the cache, so the causal mask is aligned to the bottom-right corner;
    # is_causal=True aligns it to the top-left and would let a single decoding query see only the first key.
    q_len, kv_len = q.shape[2], k_cache.shape[2]
    if q_len == 1:
        return F.scaled_dot_product_attention(q, k_cache, v_cache)
    if q_len == kv_len:
        return F.scaled_dot_product_attention(q, k_cache, v_cache, is_causal=True)
    mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device).tril(kv_len - q_len)
    return F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask=mask)


def _rotate_half(t: torch.Tensor) -> torch.Tensor:
    t0, t1 = t.chunk(2, dim=-1)
    return torch.cat((-t1, t0), dim=-1)


class MockInferState:
    def __init__(self, batch_size: int, max_len_in_batch: int, seq_lens: List[int]):
        self.batch_size = batch_size
        self.max_len_in_batch = max_len_in_batch
        self.b_seq_len = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
        self.b_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")

        class ReqManager:
            def __init__(self, bsz: int, max_len: int):
                self.req_to_token_indexs = torch.arange(bsz * max_len, dtype=torch.int32, device="cuda").view(bsz, max_len)

        self.req_manager = ReqManager(batch_size, max_len_in_batch)
        # cache row of each request's current (decoding) token, i.e. its last position
        self.mem_index = self.req_manager.req_to_token_indexs[
            self.b_req_idx.long(), self.b_seq_len.long() - 1
        ].contiguous()


class Int8AttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        rank_ratio: float,
        device: str,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        if hidden_size != num_heads * head_dim:
            raise ValueError("hidden_size must equal num_heads * head_dim")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rank_head = max(1, int(rank_ratio * head_dim))

        self.down_proj_q = nn.Linear(hidden_size, num_heads * self.rank_head, bias=False, device=device, dtype=dtype)
        # latent K and V of the current token in one GEMM: [K latents of all heads | V latents of all heads]
        self.down_proj_kv = nn.Linear(hidden_size, 2 * num_heads * self.rank_head, bias=False, device=device, dtype=dtype)
        self.up_proj_q = nn.Parameter(torch.empty(self.rank_head, hidden_size, device=device, dtype=dtype))
        self.up_proj_k = nn.Parameter(torch.empty(self.rank_head, hidden_size, device=device, dtype=dtype))
        self.up_proj_v = nn.Parameter(torch.empty(self.rank_head, hidden_size, device=device, dtype=dtype))
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.input_layernorm_weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

        torch.nn.init.xavier_uniform_(self.down_proj_q.weight)
        torch.nn.init.xavier_uniform_(self.down_proj_kv.weight)
        torch.nn.init.xavier_uniform_(self.up_proj_q)
        torch.nn.init.xavier_uniform_(self.up_proj_k)
        torch.nn.init.xavier_uniform_(self.up_proj_v)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)

        wuk_q, wuk_scale = self._quantize_wuk(self.up_proj_k.detach())
        self.register_buffer("wuk_q", wuk_q, persistent=False)
        self.register_buffer("wuk_scale", wuk_scale, persistent=False)
        self.register_buffer("wuvout", self._compose_wuvout().detach(), persistent=False)

    def _compose_wuvout(self) -> torch.Tensor:
        wuv = self.up_proj_v.view(self.num_heads, self.rank_head, self.head_dim)
        wout = self.o_proj.weight.t().contiguous().view(self.num_heads, self.head_dim, self.hidden_size)
        return (wuv @ wout).reshape(self.num_heads * self.rank_head, self.hidden_size).contiguous()

    def _quantize_wuk(self, wuk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = wuk.abs().amax(dim=0, keepdim=True) / 128
        scale = scale.clamp_min(1e-4).to(torch.float16)
        wuk_q = torch.clamp(torch.round(wuk / scale), -128, 127).to(torch.int8)
        return wuk_q.contiguous(), scale.contiguous()

    def forward(
        self,
        x: torch.Tensor,
        cache_latent_k: torch.Tensor,
        cache_latent_k_scale: torch.Tensor,
        cache_latent_v: torch.Tensor,
        cache_latent_v_scale: torch.Tensor,
        infer_state: MockInferState,
    ) -> torch.Tensor:
        bsz, q_len, _ = x.shape
        if q_len != 1:
            raise ValueError(f"Int8 latent attention only supports decode query_len=1, got {q_len}")
        x = _rms_norm(x, self.input_layernorm_weight)
        residual = x

        latent_q = self.down_proj_q(x).view(bsz, q_len, self.num_heads, self.rank_head)
        wuq = self.up_proj_q.view(self.num_heads, self.rank_head, self.head_dim)
        q = (latent_q.transpose(1, 2) @ wuq).transpose(1, 2).contiguous()
        q = q.view(bsz * q_len, self.num_heads, self.head_dim)

        # quantize the current token's latent K and V and write them into its cache row, which the attention reads
        latent_kv = self.down_proj_kv(x).view(bsz * q_len, 2, self.num_heads, self.rank_head)
        destindex_copy_quantize_latent(latent_kv[:, 0], infer_state.mem_index, cache_latent_k, cache_latent_k_scale)
        destindex_copy_quantize_latent(latent_kv[:, 1], infer_state.mem_index, cache_latent_v, cache_latent_v_scale)

        attn_output = token_decode_attention_flash_int8_latent_per_head_decoding_rope(
            q,
            infer_state,
            self.num_heads,
            self.head_dim,
            cache_latent_k,
            cache_latent_k_scale,
            cache_latent_v,
            cache_latent_v_scale,
            self.wuk_q,
            self.wuk_scale,
        )
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.rank_head)
        return attn_output @ self.wuvout + residual


class FP16LayerBaseline(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        head_dim: int,
        device: str,
        dtype: torch.dtype = torch.float16,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        if hidden_size != num_heads * head_dim:
            raise ValueError("hidden_size must equal num_heads * head_dim")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        # RoPE as in the int8 latent attention kernel (rotate-half pairs (i, i + head_dim / 2))
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)

        self.ffn_up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.ffn_gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.ffn_down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.input_layernorm_weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.post_attention_layernorm_weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def _project_attention_inputs(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # k_cache / v_cache are preallocated with q_len free slots at the end, where the new keys and values are
        # written in place.
        bsz, q_len, _ = x.shape
        x = _rms_norm(x, self.input_layernorm_weight)
        residual = x

        # RoPE of the new queries and keys at their positions kv_len - q_len, ..., kv_len - 1; the cached keys already
        # carry theirs
        kv_len = k_cache.shape[2]
        pos = torch.arange(kv_len - q_len, kv_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)[:, None, :]
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)

        q = self.q_proj(x).view(bsz, -1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, -1, self.num_heads, self.head_dim)
        q = (q * cos + _rotate_half(q) * sin).transpose(1, 2).contiguous()
        k = (k * cos + _rotate_half(k) * sin).transpose(1, 2)
        v = self.v_proj(x).view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        k_cache[:, :, -q_len:] = k
        v_cache[:, :, -q_len:] = v
        return q, k_cache, v_cache, residual

    def attention_only_with_backend(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        backend: SDPBackend,
    ) -> torch.Tensor:
        q, k_cache, v_cache, residual = self._project_attention_inputs(x, k_cache, v_cache)
        bsz = x.shape[0]
        with sdpa_kernel(backend):
            attn_output = _cached_sdpa(q, k_cache, v_cache)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, -1, self.hidden_size)
        return self.o_proj(attn_output) + residual

    def attention_only(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        return self.attention_only_with_backend(x, k_cache, v_cache, SDPBackend.MATH)

    def attention_only_flash(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        q, k_cache, v_cache, residual = self._project_attention_inputs(x, k_cache, v_cache)
        bsz = x.shape[0]
        attn_output = _cached_sdpa(q, k_cache, v_cache)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, -1, self.hidden_size)
        return self.o_proj(attn_output) + residual

    def ffn_only(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = _rms_norm(x, self.post_attention_layernorm_weight)
        ffn_up = self.ffn_up_proj(x)
        ffn_gate = self.ffn_gate_proj(x)
        mid = ffn_up * F.silu(ffn_gate)
        return self.ffn_down_proj(mid) + residual

    def forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        return self.ffn_only(self.attention_only(x, k_cache, v_cache))

