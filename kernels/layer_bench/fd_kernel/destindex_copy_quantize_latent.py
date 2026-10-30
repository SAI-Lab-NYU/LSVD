import torch
import triton
import triton.language as tl


@triton.jit
def _destindex_copy_quantize_latent_kernel(
    Latent, Dest_loc, Cache, Cache_scale,
    stride_l_bs, stride_l_h, stride_l_r,
    stride_c_bs, stride_c_h, stride_c_r,
    stride_s_bs, stride_s_h,
    RANK: tl.constexpr,
    RANK_PAD: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_r = tl.arange(0, RANK_PAD)
    mask_r = offs_r < RANK

    dest_index = tl.load(Dest_loc + cur_index).to(tl.int64)
    latent = tl.load(
        Latent + cur_index * stride_l_bs + cur_head * stride_l_h + offs_r * stride_l_r, mask=mask_r, other=0.0
    ).to(tl.float32)

    # per-(token, head) scale amax / 128, as the cache built by the benchmark
    scale = tl.maximum(tl.max(tl.abs(latent), axis=0) / 128.0, 1e-4).to(tl.float16)
    scaled = latent / scale.to(tl.float32)
    rounded = tl.where(scaled >= 0, scaled + 0.5, scaled - 0.5).to(tl.int32)
    q = tl.maximum(tl.minimum(rounded, 127), -128).to(tl.int8)

    tl.store(Cache + dest_index * stride_c_bs + cur_head * stride_c_h + offs_r * stride_c_r, q, mask=mask_r)
    tl.store(Cache_scale + dest_index * stride_s_bs + cur_head * stride_s_h, scale)


@torch.no_grad()
def destindex_copy_quantize_latent(latent, dest_loc, cache, cache_scale):
    # latent: (tokens, head_num, rank) -> int8 rows dest_loc of cache (batch*seqlen, head_num, rank),
    # with the token-wise scale in cache_scale (batch*seqlen, head_num)
    token_num, head_num, rank = latent.shape
    assert cache.shape[1:] == (head_num, rank)
    assert cache_scale.shape[1] == head_num
    _destindex_copy_quantize_latent_kernel[(token_num, head_num)](
        latent, dest_loc, cache, cache_scale,
        latent.stride(0), latent.stride(1), latent.stride(2),
        cache.stride(0), cache.stride(1), cache.stride(2),
        cache_scale.stride(0), cache_scale.stride(1),
        RANK=rank,
        RANK_PAD=triton.next_power_of_2(rank),
        num_warps=1,
    )
