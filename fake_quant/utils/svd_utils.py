import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import os
from . import (
    hadamard_utils,
    rotation_utils,
    model_utils,
    quant_utils,
    utils,
    act_aware_utils,
    loss_aware_utils,
    data_utils,
    grad_info_utils,
    qwen_vlm_utils,
    molmo_utils,
    reasoning_calib_utils,
)
import logging

import os
import torch.distributed as dist
import datetime
import math

def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size


def _match_module_weight_dtype(module, x):
    if hasattr(module, "quantizer") and hasattr(module, "module"):
        return x
    weight = getattr(module, "weight", None)
    if (
        weight is not None
        and x.is_floating_point()
        and weight.is_floating_point()
        and x.dtype != weight.dtype
    ):
        return x.to(weight.dtype)
    return x


def _is_qwen3_vl_model(args, model, tokenizer=None):
    return qwen_vlm_utils.resolve_qwen_vlm_family(args=args, model=model, tokenizer=tokenizer) == "qwen3vl"

# Smooth truncate selection across layers by capping per-layer selections.
def smooth_truncate_selection(top_indices, top_scores, num_layers, keys=None, max_factor=2.0, is_per_head_svd=False, preserve_total=True):
    if keys is None:
        keys = list(top_indices.keys())
    if is_per_head_svd:
        layer_indices_dict = {}
        for key in keys:
            if key not in top_indices:
                continue
            for item in top_indices[key]:
                if len(item) == 3:
                    layer, sv_idx, head_idx = item
                else:
                    layer, sv_idx = item
                    head_idx = None
                if layer not in layer_indices_dict:
                    layer_indices_dict[layer] = {}
                if key not in layer_indices_dict[layer]:
                    layer_indices_dict[layer][key] = {} if head_idx is not None else []
                if head_idx is not None:
                    if head_idx not in layer_indices_dict[layer][key]:
                        layer_indices_dict[layer][key][head_idx] = []
                    layer_indices_dict[layer][key][head_idx].append(sv_idx)
                else:
                    layer_indices_dict[layer][key].append(sv_idx)
        return top_indices, top_scores, layer_indices_dict

    new_top_indices = {}
    new_top_scores = {}
    new_layer_indices_dict = {}

    for key in keys:
        if key not in top_indices or key not in top_scores:
            continue
        items = list(zip(top_indices[key], top_scores[key]))
        if len(items) == 0:
            new_top_indices[key] = []
            new_top_scores[key] = []
            continue

        per_layer = {}
        for (layer, sv_idx), score in items:
            if layer not in per_layer:
                per_layer[layer] = []
            per_layer[layer].append((sv_idx, score))

        total_count = len(items)
        mean_per_layer = total_count / max(1, num_layers)
        cap = max(1, int(math.ceil(mean_per_layer * max_factor)))

        all_items_sorted = sorted(
            [((layer, sv_idx), score) for (layer, sv_idx), score in items],
            key=lambda x: x[1], reverse=True
        )

        kept = []
        kept_set = set()
        layer_counts = {}
        for (layer, sv_idx), score in all_items_sorted:
            if layer_counts.get(layer, 0) < cap:
                kept.append((layer, sv_idx, score))
                kept_set.add((layer, sv_idx))
                layer_counts[layer] = layer_counts.get(layer, 0) + 1

        if preserve_total and len(kept) < total_count:
            deficit = total_count - len(kept)
            remaining = [((layer, sv_idx), score) for ((layer, sv_idx), score) in all_items_sorted if (layer, sv_idx) not in kept_set]
            target_cap = cap
            idx = 0
            while deficit > 0 and idx < len(remaining):
                progressed = False
                for j in range(idx, len(remaining)):
                    (layer, sv_idx), score = remaining[j]
                    if layer_counts.get(layer, 0) < target_cap:
                        kept.append((layer, sv_idx, score))
                        kept_set.add((layer, sv_idx))
                        layer_counts[layer] = layer_counts.get(layer, 0) + 1
                        deficit -= 1
                        progressed = True
                        remaining[j] = None
                        if deficit == 0:
                            break
                if deficit == 0:
                    break
                if not progressed:
                    target_cap += 1
                remaining = [r for r in remaining if r is not None]
                idx = 0

        for layer, sv_idx, _ in kept:
            if layer not in new_layer_indices_dict:
                new_layer_indices_dict[layer] = {}
            if key not in new_layer_indices_dict[layer]:
                new_layer_indices_dict[layer][key] = []
            new_layer_indices_dict[layer][key].append(sv_idx)

        kept_sorted = sorted(kept, key=lambda x: x[2], reverse=True)
        new_top_indices[key] = [(layer, sv_idx) for (layer, sv_idx, _) in kept_sorted]
        new_top_scores[key] = [score for (_, _, score) in kept_sorted]

    return new_top_indices, new_top_scores, new_layer_indices_dict

# Add global variables to track the sum of ranks
total_rank_sum = 0
total_linear_count = 0


def _copy_cached_calibration_stats(source_module, target_module, module_name=None):
    if source_module is None or target_module is None:
        return target_module
    for attr in ["scaling_diag_matrixS", "output_grad_scaling_matrix"]:
        if hasattr(source_module, attr):
            value = getattr(source_module, attr)
            if value is not None:
                setattr(target_module, attr, value.detach().cpu() if isinstance(value, torch.Tensor) else value)
    if module_name is not None:
        setattr(target_module, "gptsvd_source_name", module_name)
    return target_module


class SVDLinear(nn.Module):
    def __init__(self, U, S, V, bias=None, sigma_fuse="UV", had_K=None, K=-1, had_mode='hadamard', is_ma_hack=False, original_linear=None, topk=1, is_per_head_svd=False, rank_split=None) -> None:
        super().__init__()
        self.rank_split = rank_split
        if sigma_fuse == 'local_ft' and S is not None:
            raise ValueError(
                "local_ft SVDLinear expects already fused local-ft/QAT A/B weights with S=None; "
                "got non-None S, which usually means raw SVD info was saved after local-ft/QAT failed."
            )
        if is_per_head_svd:
            if bias is not None:
                self.bias = nn.Parameter(bias.contiguous())
            else:
                self.register_parameter("bias", None)
            if rank_split is not None:
                self.head_num = len(V)
                self.ALinear = nn.ParameterList()
                if sigma_fuse == "UV":
                    V_ = []
                    for i in range(self.head_num):
                        self.ALinear.append(nn.Parameter(U[i].mul(S[i].sqrt()).contiguous()))
                        V_.append(V[i].t().mul(S[i].sqrt().view(-1, 1)).contiguous())
                    V_ = torch.cat(V_, dim=0) # [Rank_sum, C_in]
                    self.BLinear = nn.Linear(V_.size(0), V_.size(1), bias=False)
                    self.BLinear.weight.data = V_
                    del V_
                elif sigma_fuse == "U":
                    for i in range(self.head_num):
                        self.ALinear.append(nn.Parameter(U[i].mul(S[i]).contiguous()))
                    V_ = torch.cat(V, dim=-1).t() # [Rank_sum, C_in]
                    self.BLinear = nn.Linear(V_.size(0), V_.size(1), bias=False)
                    self.BLinear.weight.data = V_
                    del V_
                elif sigma_fuse == "V":
                    V_ = []
                    for i in range(self.head_num):
                        self.ALinear.append(nn.Parameter(U[i].contiguous()))
                        V_.append(V[i].t().mul(S[i].view(-1, 1)).contiguous())
                    V_ = torch.cat(V_, dim=0) # [Rank_sum, C_in]
                    self.BLinear = nn.Linear(V_.size(0), V_.size(1), bias=False)
                    self.BLinear.weight.data = V_
                    del V_
                elif sigma_fuse == 'local_ft':
                    V_ = []
                    for i in range(self.head_num):
                        u_i = U[i].contiguous()
                        v_i = V[i].contiguous()
                        rank_i = u_i.shape[-1]
                        if v_i.shape[0] != rank_i:
                            raise ValueError(
                                f"local_ft per-head V/Blinear must have shape (rank, hidden); "
                                f"got {tuple(v_i.shape)} for rank {rank_i}. "
                                "This usually means raw SVD V was used after local-ft/QAT failed."
                            )
                        self.ALinear.append(nn.Parameter(u_i))
                        V_.append(v_i)
                    V = torch.cat(V_, dim=0)
                    del V_
                    S = None
                    del S
                    self.BLinear = nn.Linear(V.size(1), V.size(0), bias=False)
                    self.BLinear.weight.data = V
                else:
                    raise ValueError(f"Unsupported sigma_fuse in per head SVD: {sigma_fuse}")
            else:
                if sigma_fuse == 'local_ft':
                    rank = U.size(-1)
                    if V.size(1) != rank:
                        raise ValueError(
                            f"local_ft per-head V/Blinear must have shape (heads, rank, hidden); "
                            f"got {tuple(V.shape)} for rank {rank}. "
                            "This usually means raw SVD V was used after local-ft/QAT failed."
                        )
                    V = V.contiguous()
                    self.BLinear = nn.Linear(V.size(2), V.size(0)* V.size(1), bias=False)
                    # U: n, c, c(r)
                    # V:  n, c(r),C_in
                else:
                    self.BLinear = nn.Linear(V.size(1), V.size(2) * V.size(0), bias=False) # c_in, rank * n
                    # U: n, c, rank (c*n = C_out)
                    # S: n, rank 
                    # V: n, C_in, rank
                self.head_num = V.size(0)
                self.head_rank  = V.size(2)
                if sigma_fuse == "UV":
                    self.ALinear = nn.Parameter(U.mul(S.sqrt().unsqueeze(1)).contiguous())
                    self.BLinear.weight.data = V.mul(S.sqrt().unsqueeze(1)).transpose(1, 2).reshape(-1, V.size(1)).contiguous()
                elif sigma_fuse == "U":
                    self.ALinear = nn.Parameter(U.mul(S.unsqueeze(1)).contiguous())
                    self.BLinear.weight.data = V.transpose(1, 2).reshape(-1, V.size(1)).contiguous()
                elif sigma_fuse == "V":
                    self.ALinear = nn.Parameter(U.contiguous())
                    self.BLinear.weight.data = V.mul(S.unsqueeze(1)).transpose(1, 2).reshape(-1, V.size(1)).contiguous()
                elif sigma_fuse == "local_ft":
                    self.head_rank = V.size(1)
                    self.ALinear = nn.Parameter(U.contiguous())
                    self.BLinear.weight.data = V.reshape(-1, V.size(2)).contiguous()
                else:
                    raise ValueError(f"Unsupported sigma_fuse in per head SVD: {sigma_fuse}")
        else:
            self.ALinear = nn.Linear(U.size(1), U.size(0), bias=bias is not None)
            if bias is not None:
                self.ALinear.bias.data = bias
            if sigma_fuse == 'local_ft':
                self.BLinear = nn.Linear(V.size(1), V.size(0), bias=False)
            else:
                self.BLinear = nn.Linear(V.size(0), V.size(1), bias=False)
            if sigma_fuse == "UV":
                self.ALinear.weight.data = U.mul(S.sqrt()).contiguous()
                self.BLinear.weight.data = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
            elif sigma_fuse == "U":
                self.ALinear.weight.data = U.mul(S).contiguous()
                self.BLinear.weight.data = V.t().contiguous()
            elif sigma_fuse == "V":
                self.ALinear.weight.data = U.contiguous()
                self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()
            elif sigma_fuse == "adaptive":
                eps = 1e-6
                scale = U.abs().max(dim=0).values/ (V.abs().max(dim=0).values + eps) # [c, r] -> [r]
                self.ALinear.weight.data = U.mul(S**(1/(scale+1))).contiguous()
                self.BLinear.weight.data = V.t().mul(S**(scale/(scale+1)).view(-1, 1)).contiguous()
            elif sigma_fuse == 'profile':
                self.ALinear.weight.data = U.mul(S).contiguous()
                self.BLinear.weight.data = V.t().contiguous()
                self.S = S
                self.U = U
                self.V = V
            elif sigma_fuse == 'local_ft':
                self.ALinear.weight.data = U
                self.BLinear.weight.data = V
            elif float(sigma_fuse) <=1.0 and float(sigma_fuse)>=0:
                self.ALinear.weight.data = U.mul(S**(1-float(sigma_fuse))).contiguous()
                self.BLinear.weight.data = V.t().mul((S**float(sigma_fuse)).view(-1, 1)).contiguous()
            else:
                raise RuntimeError(f"Error: unsupported sigma mode {sigma_fuse}")
        self.had_K = had_K
        self.K = K
        self.had_mode = had_mode

        # Add for collecting latent distribution attributes
        self.collect_latent = False
        self.alinear_hook_handle = None
        self.latent_stats = None
        
        # Add MA hack attributes
        self.is_ma_hack = is_ma_hack
        self.original_linear = original_linear
        self.topk = topk
        self.is_per_head_svd = is_per_head_svd

    def get_topk_location(self, data, k=1):
        """
        Get top-k locations and values from data tensor.
        Args:
            data (torch.Tensor): shape [bsz, n, c] or [bsz, c]
            k (int): top-k value to retrieve (default = 1). If k < 0, returns |k| random locations instead.

        Returns:
            locations (dict): keys are str(rank), values are tensors of shape [bsz, |k|, 2] with (n_idx, c_idx)
            values (dict): keys are str(rank), values are tensors of shape [bsz, |k|]
        """
        if data.dim() == 2:
            # Handle case where data is [bsz, c] (no sequence dimension)
            bsz, c = data.shape
            n = 1
            data = data.unsqueeze(1)  # [bsz, 1, c]
        else:
            bsz, n, c = data.shape
            
        flat = data.view(bsz, -1)  # shape [bsz, n * c]
        
        # Handle negative k values - use random sampling instead of top-k
        if k < 0:
            abs_k = abs(k)
            # Generate random indices for each batch item
            total_elements = flat.shape[1]  # n * c
            random_inds = torch.randint(0, total_elements, (bsz, abs_k), device=flat.device)
            
            # Get values at random locations
            batch_indices = torch.arange(bsz, device=flat.device).unsqueeze(1).expand(-1, abs_k)
            random_vals = flat[batch_indices, random_inds]
            
            # Convert flat index to (n_idx, c_idx)
            n_idx = random_inds // c  # shape [bsz, abs_k]
            c_idx = random_inds % c   # shape [bsz, abs_k]
        else:
            # Original top-k logic
            topk_vals, topk_inds = torch.topk(flat, k, dim=-1)  # shape [bsz, k]
            random_vals = topk_vals
            n_idx = topk_inds // c  # shape [bsz, k]
            c_idx = topk_inds % c   # shape [bsz, k]

        # Stack to get (n_idx, c_idx) for each location
        locations = torch.stack([n_idx, c_idx], dim=-1)  # shape [bsz, |k|, 2]

        num_locations = abs(k) if k < 0 else k
        out_locs = {str(rank): locations[:, rank] for rank in range(num_locations)}
        out_vals = {str(rank): random_vals[:, rank] for rank in range(num_locations)}

        return out_locs, out_vals

    def apply_had_rank(self):
        had_K = self.had_K
        K = self.K
        had_mode = self.had_mode

        if K >0 and had_mode in ['rh', 'random']:
            W = self.ALinear # input
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(W_, had_K).to(device="cpu", dtype=dtype)
            W = self.BLinear # output
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(had_K.T, W_).to(device="cpu", dtype=dtype)
            if W.bias is not None:
                b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
                W.bias.data = torch.matmul(had_K.T, b).to(device="cpu", dtype=dtype)
        elif K>0 and had_mode == 'hadamard':
            hadamard_utils.apply_exact_had_to_linear(self.ALinear, had_dim=-1, output=False)
            hadamard_utils.apply_exact_had_to_linear(self.BLinear, had_dim=-1, output=True)

    def start_collecting_latent(self):
        """Start collecting the input distribution of ALinear"""
        self.collect_latent = True
        
        # Initialize statistics
        self.latent_stats = torch.zeros(self.ALinear.in_features, device=utils.get_dev())
        
        # Register forward hook
        def hook(module, input, output):
            if not self.collect_latent:
                return
            
            # Use abs_max method
            abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
            self.latent_stats = torch.where(
                abs_max > self.latent_stats,
                abs_max,
                self.latent_stats,
            )
        
        self.alinear_hook_handle = self.ALinear.register_forward_hook(hook)
        
    def stop_collecting_latent(self):
        """Stop collecting the input distribution of ALinear"""
        self.collect_latent = False
        
        # Remove hook
        if hasattr(self, 'alinear_hook_handle') and self.alinear_hook_handle is not None:
            self.alinear_hook_handle.remove()
            self.alinear_hook_handle = None
    
    def get_latent_stats(self):
        """Get statistics of ALinear input"""
        if self.latent_stats is not None:
            return self.latent_stats.cpu()
        return None

    def apply_latent_smooth(self, alpha=0.5, eps=1e-6):
        if self.latent_stats is None or torch.all(self.latent_stats == 0):
            logging.info("No available latent space statistics, cannot smooth.")
            return False
            
        # Get compute device and original device
        compute_device = utils.get_dev()
        original_device = self.ALinear.weight.device
        original_dtype = self.ALinear.weight.dtype
        
        # Move statistics and weights to compute device
        latent_stats = self.latent_stats.to(compute_device)
        a_weight = self.ALinear.weight.data.to(compute_device)
        b_weight = self.BLinear.weight.data.to(compute_device)
        
        # Compute column-wise statistics of ALinear weights (corresponding to each latent dimension)
        weight_stats = a_weight.abs().amax(dim=0)  # max absolute value per column [rank]?

        # Compute scaling factors to ensure similar importance for each latent dimension
        # Use the reciprocal of the max absolute value as the scaling factor
        scale_factors = (latent_stats**alpha)/(weight_stats**(1-alpha))
        scale_factors = torch.clamp(scale_factors, min=eps)
        
        # Apply scaling factors to ALinear and BLinear weights
        # The input dimension of ALinear corresponds to the latent space, so scale by column
        a_weight_scaled = a_weight * scale_factors.view(1, -1) # [c, rank]
        
        # The output dimension of BLinear corresponds to the latent space, so scale by row
        # Also need to take the reciprocal of the scaling factor to keep the overall transformation unchanged
        inv_scale_factors = 1.0 / scale_factors 
        b_weight_scaled = b_weight * inv_scale_factors.view(-1, 1) # [rank, c]
        
        # Move results back to original device and dtype
        self.ALinear.weight.data = a_weight_scaled.to(device=original_device, dtype=original_dtype)
        self.BLinear.weight.data = b_weight_scaled.to(device=original_device, dtype=original_dtype)
        
        logging.info(f"Successfully smoothed latent space weights, scale factor range: {scale_factors.min().item():.4f} - {scale_factors.max().item():.4f}")
        return True

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        act_aware=False,
        loss_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1.,
        sigma_fuse="UV",
        rank_align=1.,
        had_rank=False,
        had_mode='hadamard',
        seed=0,
        module_name='o_proj',
        is_ma_hack=False,
        is_per_head_svd=False,
        is_q_headnum=False,
        num_heads=None,
        num_kv_heads=None,
        head_count_override=None,
        topk=1,
        svd_debug_store=None,
        svd_debug_key=None,
    ):
        N_heads = head_count_override if head_count_override is not None else (num_heads if is_q_headnum else num_kv_heads)
        # Ensure param_ratio is a Python float, not a tensor
        if isinstance(param_ratio, torch.Tensor):
            param_ratio = float(param_ratio.item())
        param_ratio = min(param_ratio, 2)
        
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        assert ic_split == 1 or oc_split == 1
        if is_per_head_svd:
            assert N_heads is not None and N_heads > 0, (
                f"Invalid head count for per-head SVD on {module_name}: {N_heads}"
            )
            assert linear.out_features % N_heads == 0, (
                f"Per-head SVD reshape mismatch for {module_name}: "
                f"out_features={linear.out_features}, head_count={N_heads}"
            )
            compressed_params = int(n_params/N_heads * param_ratio)
            rank = compressed_params // (linear.in_features + linear.out_features/N_heads)
        else:
            rank = compressed_params // (linear.in_features + linear.out_features)
        # rank align
        rank = int(np.ceil(rank / rank_align) * rank_align)
        
        # Update total rank sum and linear layer count
        total_rank_sum[module_name] += rank
        total_linear_count[module_name] += 1

        if had_rank:
            utils.set_seed(seed)
            if had_mode == 'hadamard':
                K = 1
                had_K = None
            elif had_mode == 'rh':
                had_K = rotation_utils.get_orthogonal_matrix(rank, 'hadamard')
                K = rank
            elif had_mode == 'random':
                had_K = rotation_utils.get_orthogonal_matrix(rank, 'random')
                K = rank
        # Correctly move weights to CUDA device
        if is_per_head_svd:
            w = linear.weight.data.view(N_heads, -1, linear.in_features).float().to(utils.get_dev())
        else:
            w = linear.weight.data.float().to(utils.get_dev())
        if act_aware or loss_aware:  # activation whitening
            scaling_diag_matrix = torch.ones(linear.in_features, device=utils.get_dev())  # avoid zero division
            scaling_matrix_inv = None
            if hasattr(linear, "scaling_diag_matrix"):
                scaling_diag_matrix *= linear.scaling_diag_matrix.to(utils.get_dev())**alpha
                scaling_diag_matrix += 1e-6  # avoid zero division
                scaling_matrix_inv = None
                if is_per_head_svd:
                    w = w * scaling_diag_matrix.view(1, 1, -1)
                else:
                    w = w * scaling_diag_matrix.view(1, -1)
            elif hasattr(linear, "scaling_diag_matrixS"):
                scaling_diag_matrix = linear.scaling_diag_matrixS.to(utils.get_dev())
                try:
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                except Exception as e:
                    logging.info("Warning: scaling_diag_matrix is not full rank!")
                    scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                w = w.double() @ scaling_diag_matrix.double()
        if loss_aware:
            output_grad_scaling_matrix = linear.output_grad_scaling_matrix.to(utils.get_dev())
            if is_per_head_svd:
                w = output_grad_scaling_matrix.double() @ w.double()
            else:
                w = output_grad_scaling_matrix.double() @ w.double()
            output_grad_scaling_matrix_inv = torch.linalg.inv(output_grad_scaling_matrix.double())
        Us = []
        Ss = []
        Vs = []
        try:
            if loss_aware:
                logging.info(f"Rank: {rank} for {module_name} with loss_aware")
            elif act_aware:
                logging.info(f"Rank: {rank} for {module_name} with act_aware")
            else:
                logging.info(f"Rank: {rank}")
            # SVD decomposition
            # U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False)
            U, S, Vt = torch.linalg.svd(w.to(torch.float64), full_matrices=False)

            if svd_debug_store is not None and svd_debug_key:
                svd_debug_store[svd_debug_key] = S.detach().cpu().clone()
            
            # Low rank approximation to the target rank
            if is_per_head_svd:
                U = U[:, :, :rank]
                S = S[:, :rank]
                Vt = Vt[:, :rank, :]
                V = Vt.transpose(1, 2)
            else:
                U = U[:, :rank]
                S = S[:rank]
                Vt = Vt[:rank, :]
                V = Vt.T
        except Exception as e:
            logging.info(f"SVD failed for {linear}: {e}")
            return nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
        if act_aware or loss_aware:  # activation whitening
            if scaling_matrix_inv is None:
                if is_per_head_svd:
                    V = V / scaling_diag_matrix.view(1, -1, 1)
                else:
                    V = V / scaling_diag_matrix.view(-1, 1)
            else:
                if is_per_head_svd:
                    V =  scaling_matrix_inv.T.double().unsqueeze(0).expand(N_heads, -1, -1) @ V # V: n, C_in, rank, scaling_matrix_inv: C_in, C_in
                else:
                    V =  scaling_matrix_inv.T.double() @ V
        if loss_aware:
            if is_per_head_svd:
                U = output_grad_scaling_matrix_inv.double() @ U
            else:
                U = output_grad_scaling_matrix_inv.double() @ U
        Us = [U]
        Ss = [S]
        Vs = [V]

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # nan or inf check
        for S in Ss:
            if (S != S).any():
                logging.info("nan in S")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )
        for U in Us:
            if (U != U).any():
                logging.info("nan in U")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )
        for V in Vs:
            if (V != V).any():
                logging.info("nan in V")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )

        assert len(Us) == len(Ss) == len(Vs) == 1
        
        if had_rank:
            if is_per_head_svd:
                logging.info("Warning: head wise svd is not supported for had_rank yet; fallback to non-had_rank path.")
                had_rank = False
            if had_rank:
                new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse, had_K, K, had_mode, is_ma_hack, linear if is_ma_hack else None, topk)
            else:
                new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear if is_ma_hack else None, topk=topk, is_per_head_svd=is_per_head_svd)
        else:
            if is_per_head_svd:
                new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear if is_ma_hack else None, topk=topk, is_per_head_svd=is_per_head_svd)
            else:
                new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear if is_ma_hack else None, topk=topk)
        _copy_cached_calibration_stats(linear, new_linear, module_name=module_name)
        new_linear.to(linear.weight.dtype)
        return new_linear.cpu()


    @staticmethod
    def from_linearkv(
        linear: nn.Linear,
        linear1: nn.Linear,
        param_ratio: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1.,
        sigma_fuse="UV",
        rank_align=1.,
        had_rank=False,
        had_mode='hadamard',
        seed=0,
        module_name='up_proj',
        true_param_ratio=False,
        is_ma_hack=False,
        topk=1,
    ):
        
        # Ensure param_ratio is a Python float, not a tensor
        if isinstance(param_ratio, torch.Tensor):
            param_ratio = float(param_ratio.item())
        param_ratio = min(param_ratio, 2) # change the ratio here
        
        n_params = linear.weight.numel()
        if true_param_ratio:
            compressed_params = int(n_params * param_ratio * 2)
            assert ic_split == 1 or oc_split == 1
            rank = compressed_params // (linear.in_features + linear.out_features*2)
        else:
            compressed_params = int(n_params * param_ratio)
            assert ic_split == 1 or oc_split == 1
            rank = compressed_params // (linear.in_features + linear.out_features)
        # rank align
        rank = int(np.ceil(rank / rank_align) * rank_align)
        
        # Update total rank sum and linear layer count (here calculated as 2 linear layers share a rank)
        total_rank_sum[module_name] += rank
        total_linear_count[module_name] += 1

        if had_rank:
            utils.set_seed(seed)
            if had_mode == 'hadamard':
                K = 1
                had_K = None
            elif had_mode == 'rh':
                had_K = rotation_utils.get_orthogonal_matrix(rank, 'hadamard')
                K = rank
            elif had_mode == 'random':
                had_K = rotation_utils.get_orthogonal_matrix(rank, 'random')
                K = rank
                
        # Correctly move weights to CUDA device
        device = utils.get_dev()
        w = torch.cat([linear.weight.data.float(), linear1.weight.data.float()], dim=0).to(device)
        if act_aware:
            scaling_diag_matrix = torch.ones(linear.in_features, device=utils.get_dev())  # avoid zero division
            if hasattr(linear, "scaling_diag_matrix"):
                scaling_diag_matrix *= linear.scaling_diag_matrix.to(utils.get_dev())**alpha
                scaling_diag_matrix += 1e-6  # avoid zero division
                scaling_matrix_inv = None
                w = w * scaling_diag_matrix.view(1, -1)
            elif hasattr(linear, "scaling_diag_matrixS"):
                scaling_diag_matrix = linear.scaling_diag_matrixS.to(utils.get_dev())
                try:
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                except Exception as e:
                    logging.info("Warning: scaling_diag_matrix is not full rank!")
                    scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                w = w @ scaling_diag_matrix.float()
        Us = []
        Ss = []
        Vs = []
        try:
            logging.info(f"KVRank: {rank}")
            # SVD decomposition
            U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False)

            # Low rank approximation to the target rank
            U = U[:, :rank]
            S = S[:rank]
            Vt = Vt[:rank, :]
            V = Vt.T
        except Exception as e:
            logging.info(f"Fuse KV SVD failed for {linear}: {e}")
            logging.info(f"Matrix information: shape={w.size()}, whether contains NaN: {torch.isnan(w).any()}, whether contains Inf: {torch.isinf(w).any()}")
            logging.info(f"Matrix statistics: min={w.min().item()}, max={w.max().item()}, mean={w.mean().item()}, std={w.std().item()}")
            return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device)
                )
        if act_aware:
            if scaling_matrix_inv is None:
                V = V / scaling_diag_matrix.view(-1, 1)
            else:
                V =  scaling_matrix_inv.T.float() @ V
        U = U.view(2, -1, rank)
        Us = [U[0], U[1]]
        Ss = [S]
        Vs = [V]

        if linear.bias is not None:
            biask = linear.bias.data
            biasv = linear1.bias.data
        else:
            biask = None
            biasv = None

        # nan or inf check
        for S in Ss:
            if (S != S).any():
                logging.info("nan in S")
                # For from_linearkv method return two linear layers
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device)
                )
        for U in Us:
            if (U != U).any():
                logging.info("nan in U")
                # For from_linearkv method return two linear layers
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device)
                )
        for V in Vs:
            if (V != V).any():
                logging.info("nan in V")
                # For from_linearkv method return two linear layers
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device)
                )

        assert len(Us)/2 == len(Ss) == len(Vs) == 1
        
        if had_rank:
            new_linearK = SVDLinear(Us[0], Ss[0], Vs[0], biask, sigma_fuse, had_K, K, had_mode, is_ma_hack, linear if is_ma_hack else None, topk)
            new_linearV = SVDLinear(Us[1], Ss[0], Vs[0], biasv, sigma_fuse, had_K, K, had_mode, is_ma_hack, linear1 if is_ma_hack else None, topk)
        else:
            new_linearK = SVDLinear(Us[0], Ss[0], Vs[0], biask, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear if is_ma_hack else None, topk=topk)
            new_linearV = SVDLinear(Us[1], Ss[0], Vs[0], biasv, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear1 if is_ma_hack else None, topk=topk)
        _copy_cached_calibration_stats(linear, new_linearK, module_name=module_name)
        _copy_cached_calibration_stats(linear1, new_linearV, module_name="v_proj" if module_name == "k_proj" else module_name)
        new_linearK.to(linear.weight.dtype)
        new_linearV.to(linear.weight.dtype)
        return new_linearK.cpu(), new_linearV.cpu()
    
    @staticmethod
    def from_linearqkv(
        linear: nn.Linear,
        linear1: nn.Linear,
        linear2: nn.Linear,
        param_ratio: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1.,
        sigma_fuse="UV",
        rank_align=1.,
        had_rank=False,
        had_mode='hadamard',
        seed=0,
        module_name='k_proj',
        true_param_ratio=False,
        use_group_svd=False,
        group_ratio=0.0,
        is_ma_hack=False,
        topk=1,
        is_per_head_svd=False,
        is_q_headnum=False,
        num_heads=None,
        num_kv_heads=None,
    ):
        N_heads = num_heads if is_q_headnum  else num_kv_heads
        # Ensure param_ratio is a Python float, not a tensor
        if isinstance(param_ratio, torch.Tensor):
            param_ratio = float(param_ratio.item())
        param_ratio = min(param_ratio, 2)
        
        n_params = linear1.weight.numel()
        rank_kv = None
        if use_group_svd:
            gqa_ratio = linear.out_features // linear1.out_features # group_ratio = Q_out_features / K_out_features
            if group_ratio > 0:
                g_ratio = group_ratio
            else:
                g_ratio = gqa_ratio
            # here as we partially share the down proj SVD weight, so we use group_ratio to calculate the rank
            # details in paper: xxx Section xxxx
            rank_kv = int(param_ratio * linear1.out_features * (gqa_ratio ** 2 + 2 * gqa_ratio) / (2 * gqa_ratio * g_ratio + 2))
            if is_per_head_svd:
                rank_kv = int(param_ratio * linear1.out_features * (gqa_ratio ** 2 + 2 * gqa_ratio) / (2 * gqa_ratio * g_ratio + 2)/N_heads)
            rank_kv = int(np.ceil(rank_kv / rank_align) * rank_align)
            rank = rank_kv * g_ratio
            
        elif true_param_ratio:
            n_params += linear.weight.numel()
            n_params += linear2.weight.numel()
            compressed_params = int(n_params * param_ratio)
            assert ic_split == 1 or oc_split == 1
            rank = compressed_params // (linear1.in_features + linear.out_features + linear1.out_features + linear2.out_features)
            if is_per_head_svd:
                compressed_params = int(n_params/N_heads * param_ratio)
                rank = compressed_params // (
                    linear1.in_features
                    + linear.out_features // N_heads
                    + linear1.out_features // N_heads
                    + linear2.out_features // N_heads
                )
        else:
            compressed_params = int(n_params * param_ratio)
            assert ic_split == 1 or oc_split == 1
            rank = compressed_params // (linear1.in_features + linear1.out_features)
            if is_per_head_svd:
                compressed_params = int(n_params/N_heads * param_ratio)
                rank = compressed_params // (linear1.in_features + linear1.out_features/N_heads)
        # rank align
        rank = int(np.ceil(rank / rank_align) * rank_align)
        
        
        # Update total rank sum and linear layer count (here calculated as 2 linear layers share a rank)
        total_rank_sum[module_name] += rank
        total_linear_count[module_name] += 1

        if had_rank: # [FIXED: add GQA had_K support]
            utils.set_seed(seed)
            if had_mode == 'hadamard':
                K = 1
                had_K = None
            elif had_mode == 'rh':
                had_K = rotation_utils.get_orthogonal_matrix(rank, 'hadamard')
                K = rank
            elif had_mode == 'random':
                had_K = rotation_utils.get_orthogonal_matrix(rank, 'random')
                K = rank
                
        # Correctly move weights to CUDA device
        device = utils.get_dev()
        if is_per_head_svd:
            w = torch.cat([linear.weight.data.view(N_heads, -1, linear.in_features).float(), linear1.weight.data.view(N_heads, -1, linear1.in_features).float(), linear2.weight.data.view(N_heads, -1, linear2.in_features).float()], dim=1).to(device) # q, k, v
            # W of shape [num_heads, 3c, in_features] # c * num_heads = out_features
        else:
            w = torch.cat([linear.weight.data.float(), linear1.weight.data.float(), linear2.weight.data.float()], dim=0).to(device) # q, k, v
        if act_aware:
            scaling_matrix_inv = None # to ensure first time run, do not load cache, so qv linear do not have scaling_diag_matrix
            scaling_diag_matrix = torch.ones(linear1.in_features, device=utils.get_dev())  # avoid zero division
            if hasattr(linear1, "scaling_diag_matrix"):
                scaling_diag_matrix *= linear1.scaling_diag_matrix.to(utils.get_dev())**alpha
                scaling_diag_matrix += 1e-6  # avoid zero division
                scaling_matrix_inv = None
                if is_per_head_svd:
                    w = w * scaling_diag_matrix.view(1, 1, -1) # w: n, 3c, C_in, scaling_diag_matrix: C_in
                else:
                    w = w * scaling_diag_matrix.view(1, -1)
            elif hasattr(linear1, "scaling_diag_matrixS"):
                scaling_diag_matrix = linear1.scaling_diag_matrixS.to(utils.get_dev())
                try:
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                except Exception as e:
                    logging.info("Warning: scaling_diag_matrix is not full rank!")
                    scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                w = w @ scaling_diag_matrix.float()
        Us = []
        Ss = []
        Vs = []
        try:
            logging.info(f"Rank: {rank}")
            logging.info(f"Rank KV: {rank_kv}") if rank_kv is not None else None
            # SVD decomposition
            U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False)

            # Low rank approximation to the target rank
            if is_per_head_svd:
                U = U[:, :, :rank]     # n, c, rank (c*n = C_out)
                S = S[:, :rank]        # n, rank 
                Vt = Vt[:, :rank, :]   # n, rank, C_in
                V = Vt.transpose(1, 2) # n, C_in, rank
            else:
                U = U[:, :rank]
                S = S[:rank]
                Vt = Vt[:rank, :]
                V = Vt.T
        except Exception as e:
            logging.info(f"Fuse QKV SVD failed for {linear}: {e}")
            logging.info(f"Matrix information: shape={w.size()}, whether contains NaN: {torch.isnan(w).any()}, whether contains Inf: {torch.isinf(w).any()}")
            logging.info(f"Matrix statistics: min={w.min().item()}, max={w.max().item()}, mean={w.mean().item()}, std={w.std().item()}")
            return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                    nn.Linear(linear2.in_features, linear2.out_features).to(linear2.weight.dtype).to(linear2.weight.device)
                )
        if act_aware:
            if scaling_matrix_inv is None:
                if is_per_head_svd:
                    V = V / scaling_diag_matrix.view(1, -1, 1) # V: n, C_in, rank, scaling_diag_matrix: C_in
                else:
                    V = V / scaling_diag_matrix.view(-1, 1)
            else:
                if is_per_head_svd:
                    V =  scaling_matrix_inv.T.float().unsqueeze(0).expand(N_heads, -1, -1) @ V # V: n, C_in, rank, scaling_matrix_inv: C_in, C_in
                else:
                    V =  scaling_matrix_inv.T.float() @ V
        # if 
        if is_per_head_svd:
            if linear.out_features == linear1.out_features:
                U = U.view(N_heads, 3, -1, rank) # n, 3c, rank (c*n = C_out)
                Us = [U[:,0,:,:], U[:,1,:,:], U[:,2,:,:]]
            else:
                U = torch.split(U, [linear.out_features//N_heads, linear1.out_features//N_heads, linear2.out_features//N_heads], dim=1)
                Us = [U[0], U[1], U[2]]
            Ss = [S]
            Vs = [V]
        else:
            if linear.out_features == linear1.out_features:
                U = U.view(3, -1, rank) # here need to set dim for GQA, not / 3
            else:
                U = torch.split(U, [linear.out_features, linear1.out_features, linear2.out_features], dim=0)
            Us = [U[0], U[1], U[2]]
            Ss = [S]
            Vs = [V]

        if linear.bias is not None:
            biasq = linear.bias.data
            biask = linear1.bias.data
            biasv = linear2.bias.data
        else:
            biasq = None
            biask = None
            biasv = None

        # nan or inf check
        for S in Ss:
            if (S != S).any():
                logging.info("nan in S")
                # For from_linearqkv method return three linear layers
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                    nn.Linear(linear2.in_features, linear2.out_features).to(linear2.weight.dtype).to(linear2.weight.device)
                )
        for U in Us:
            if (U != U).any():
                logging.info("nan in U")
                # For from_linearqkv method return three linear layers
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                    nn.Linear(linear2.in_features, linear2.out_features).to(linear2.weight.dtype).to(linear2.weight.device)
                )
        for V in Vs:
            if (V != V).any():
                logging.info("nan in V")
                # For from_linearqkv method return three linear layers
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                    nn.Linear(linear2.in_features, linear2.out_features).to(linear2.weight.dtype).to(linear2.weight.device)
                )

        assert len(Us)/3 == len(Ss) == len(Vs) == 1
        if had_rank:
            if is_per_head_svd:
                logging.info("Warning: head wise svd is not supported for had_rank yet")
            had_K_kv = had_K
            K_kv = K
            new_linearQ = SVDLinear(Us[0], Ss[0], Vs[0], biasq, sigma_fuse, had_K, K, had_mode, is_ma_hack, linear if is_ma_hack else None, topk)
            new_linearK = SVDLinear(Us[1][:,:rank_kv], Ss[0][:rank_kv], Vs[0][:,:rank_kv], biask, sigma_fuse, had_K_kv, K_kv, had_mode, is_ma_hack, linear1 if is_ma_hack else None, topk)
            new_linearV = SVDLinear(Us[2][:,:rank_kv], Ss[0][:rank_kv], Vs[0][:,:rank_kv], biasv, sigma_fuse, had_K_kv, K_kv, had_mode, is_ma_hack, linear2 if is_ma_hack else None, topk)
        else:
            if is_per_head_svd:
                new_linearQ = SVDLinear(Us[0], Ss[0], Vs[0], biasq, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear if is_ma_hack else None, topk=topk, is_per_head_svd=is_per_head_svd)
                new_linearK = SVDLinear(Us[1][:,:rank_kv], Ss[0][:rank_kv], Vs[0][:,:rank_kv], biask, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear1 if is_ma_hack else None, topk=topk, is_per_head_svd=is_per_head_svd)
                new_linearV = SVDLinear(Us[2][:,:rank_kv], Ss[0][:rank_kv], Vs[0][:,:rank_kv], biasv, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear2 if is_ma_hack else None, topk=topk, is_per_head_svd=is_per_head_svd)
            else:
                new_linearQ = SVDLinear(Us[0], Ss[0], Vs[0], biasq, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear if is_ma_hack else None, topk=topk)
                new_linearK = SVDLinear(Us[1][:,:rank_kv], Ss[0][:rank_kv], Vs[0][:,:rank_kv], biask, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear1 if is_ma_hack else None, topk=topk)
                new_linearV = SVDLinear(Us[2][:,:rank_kv], Ss[0][:rank_kv], Vs[0][:,:rank_kv], biasv, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear2 if is_ma_hack else None, topk=topk)
        new_linearQ.to(linear.weight.dtype)
        new_linearK.to(linear.weight.dtype)
        new_linearV.to(linear.weight.dtype)
        return new_linearQ.cpu(), new_linearK.cpu(), new_linearV.cpu()

    @staticmethod
    def from_linearqkv_localft(
        linear: nn.Linear,
        linear1: nn.Linear,
        linear2: nn.Linear,
        param_ratio: float,
        sigma_fuse="local_ft",
        had_rank=False,
        had_mode='random',
        seed=0,
        module_name='k_proj',
        is_per_head_svd=False,
        is_rank_allocation=False,
    ):
        device = utils.get_dev()
        svd_info = linear1.qkv_svd_info
        if sigma_fuse == "local_ft" and svd_info.get('S', None) is not None:
            raise ValueError(
                "from_linearqkv_localft expects local-ft/QAT qkv_svd_info with S=None; "
                "got non-None S, which usually means raw SVD info was saved after local-ft/QAT failed."
            )
        U = svd_info['U'].to(device)
        V = svd_info['V'].to(device)
        if is_per_head_svd:
            rank = U[0].shape[1]
            num_heads = len(U)
            if had_rank:
                logging.info("Warning: head wise svd from linear is not supporting had_rank yet")
                pass
        else:
            rank = U.shape[1]
            if had_rank:
                utils.set_seed(seed)
                if had_mode == 'hadamard':
                    K = 1
                    had_K = None
                elif had_mode == 'rh':
                    had_K = rotation_utils.get_orthogonal_matrix(rank, 'hadamard')
                    K = rank
                elif had_mode == 'random':
                    had_K = rotation_utils.get_orthogonal_matrix(rank, 'random')
                    K = rank

        # Update total rank sum and linear layer count
        # [NOTE: follow from_linearqkv calculation, we only log per head rank [rank//num_heads]]
        if not is_rank_allocation:
            total_rank_sum[module_name] += rank
            logging.info(f"Layer {module_name} uses rank: {rank}")
        total_linear_count[module_name] += 1
        if is_per_head_svd:
            Us = [[], [], []]
            if is_rank_allocation:
                rank_split = []
            else:
                rank_split = None
            for i in range(num_heads):
                # [3c, rank_] * n
                rank_ = U[i].shape[-1]
                if is_rank_allocation:
                    rank_split.append(rank_)
                    total_rank_sum[module_name] += rank_
                    logging.info(f"Layer uses rank: {rank_} on head {i}")
                if linear.out_features == linear1.out_features:
                    u_selected_ = U[i].view(3, -1, rank_)
                else:
                    u_selected_ = torch.split(U[i], [linear.out_features//num_heads, linear1.out_features//num_heads, linear2.out_features//num_heads], dim=0)
                Us[0].append(u_selected_[0])
                Us[1].append(u_selected_[1])
                Us[2].append(u_selected_[2])
            Vs = [V] # [rank_, C_in] * n
            if not is_rank_allocation:
                Us = [torch.stack(list(us), dim=0) for us in Us] # [3 *[n * [c, rank_]] -> 3 * [n, c, rank_]]
                Vs = [torch.stack(list(vs), dim=0) for vs in Vs] # [[n * [C_in, rank_]] -> [n, rank_, C_in]]
                # do not support UV of RA version, which have different rank_i
        else:
            if linear.out_features == linear1.out_features:
                U = U.view(3, -1, rank)
            else:
                U = torch.split(U, [linear.out_features, linear1.out_features, linear2.out_features], dim=0)
            Us = [U[0], U[1], U[2]]
            Vs = [V]

        if linear.bias is not None:
            biasq = linear.bias.data
            biask = linear1.bias.data
            biasv = linear2.bias.data
        else:
            biasq = None
            biask = None
            biasv = None

        if had_rank:
            if is_per_head_svd:
                logging.info("Warning: head wise svd from linear is not supporting had_rank yet")
            new_linearQ = SVDLinear(Us[0], None, Vs[0], biasq, sigma_fuse, had_K, K, had_mode)
            new_linearK = SVDLinear(Us[1], None, Vs[0], biask, sigma_fuse, had_K, K, had_mode)
            new_linearV = SVDLinear(Us[2], None, Vs[0], biasv, sigma_fuse, had_K, K, had_mode)
        else:
            if is_per_head_svd:
                new_linearQ = SVDLinear(Us[0], None, Vs[0], biasq, sigma_fuse, is_per_head_svd=is_per_head_svd, rank_split=rank_split)
                new_linearK = SVDLinear(Us[1], None, Vs[0], biask, sigma_fuse, is_per_head_svd=is_per_head_svd, rank_split=rank_split)
                new_linearV = SVDLinear(Us[2], None, Vs[0], biasv, sigma_fuse, is_per_head_svd=is_per_head_svd, rank_split=rank_split)
            else:
                new_linearQ = SVDLinear(Us[0], None, Vs[0], biasq, sigma_fuse)
                new_linearK = SVDLinear(Us[1], None, Vs[0], biask, sigma_fuse)
                new_linearV = SVDLinear(Us[2], None, Vs[0], biasv, sigma_fuse)
        _copy_cached_calibration_stats(linear, new_linearQ, module_name="q_proj")
        _copy_cached_calibration_stats(linear1, new_linearK, module_name="k_proj")
        _copy_cached_calibration_stats(linear2, new_linearV, module_name="v_proj")
        new_linearQ.to(linear.weight.dtype)
        new_linearK.to(linear.weight.dtype)
        new_linearV.to(linear.weight.dtype)
        return new_linearQ.cpu(), new_linearK.cpu(), new_linearV.cpu()
        

    @staticmethod
    def from_linear_localft(
        linear: nn.Linear,
        param_ratio: float,
        sigma_fuse="local_ft",
        had_rank=False,
        had_mode='random',
        seed=0,
        module_name='o_proj',
        is_per_head_svd=False,
        is_rank_allocation=False,
    ):
        device = utils.get_dev()
        svd_info = linear.svd_info

        # Get SVD information
        U = svd_info['U'].to(device)
        S = svd_info['S']
        V = svd_info['V'].to(device)
        if sigma_fuse == "local_ft" and S is not None:
            raise ValueError(
                "from_linear_localft expects local-ft/QAT svd_info with S=None; "
                "got non-None S, which usually means raw SVD info was saved after local-ft/QAT failed."
            )
        if is_per_head_svd:
            rank = U[0].shape[1]
            num_heads = len(U)
            if had_rank:
                logging.info("Warning: head wise svd from linear is not supporting had_rank yet")
                pass
        else:
            rank = U.shape[1]
            if had_rank:
                utils.set_seed(seed)
                if had_mode == 'hadamard':
                    K = 1
                    had_K = None
                elif had_mode == 'rh':
                    had_K = rotation_utils.get_orthogonal_matrix(rank, 'hadamard')
                    K = rank
                elif had_mode == 'random':
                    had_K = rotation_utils.get_orthogonal_matrix(rank, 'random')
                    K = rank

        # Update total rank sum and linear layer count
        if not is_rank_allocation:
            total_rank_sum[module_name] += rank
            logging.info(f"Layer {module_name} uses rank: {rank}")

        total_linear_count[module_name] += 1
        if is_per_head_svd:
            Us = []
            if is_rank_allocation:
                rank_split = []
            else:
                rank_split = None
            for i in range(num_heads):
                rank_ = U[i].shape[-1]
                if is_rank_allocation:
                    rank_split.append(rank_)
                    total_rank_sum[module_name] += rank_
                    logging.info(f"Layer uses rank: {rank_} on head {i}")
                Us.append(U[i])
            Us = [Us]
            Vs = [V]
            if not is_rank_allocation:
                Us = [torch.stack(list(us), dim=0) for us in Us] # # [n * [c, rank_] -> [n, c, rank_]
                Vs = [torch.stack(list(vs), dim=0) for vs in Vs] # [[n * [C_in, rank_]] -> [n, rank_, C_in]]
                # do not support UV of RA version, which have different rank_i
        else:
            Us = [U]
            Vs = [V]

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        if had_rank:
            if is_per_head_svd:
                logging.info("Warning: head wise svd from linear is not supporting had_rank yet")
                pass
            else:
                new_linear = SVDLinear(Us[0], None, Vs[0], bias, sigma_fuse, had_K, K, had_mode)
        else:
            if is_per_head_svd:
                new_linear = SVDLinear(Us[0], S, Vs[0], bias, sigma_fuse, is_per_head_svd=is_per_head_svd, rank_split=rank_split)
            else:
                new_linear = SVDLinear(Us[0], S, Vs[0], bias, sigma_fuse)
        _copy_cached_calibration_stats(linear, new_linear, module_name=module_name)
        new_linear.to(linear.weight.dtype)
        return new_linear.cpu()


    @staticmethod
    def from_linearkv_localft(
        linear: nn.Linear,
        linear1: nn.Linear,
        param_ratio: float,
        sigma_fuse="local_ft",
        had_rank=False,
        had_mode='random',
        seed=0,
        module_name='k_proj',
    ):
        device = utils.get_dev()
        svd_info = linear.svd_info
        if sigma_fuse == "local_ft" and svd_info.get('S', None) is not None:
            raise ValueError(
                "from_linearkv_localft expects local-ft/QAT svd_info with S=None; "
                "got non-None S, which usually means raw SVD info was saved after local-ft/QAT failed."
            )
        U = svd_info['U'].to(device)
        V = svd_info['V'].to(device)
        rank = U.shape[1]
        if had_rank:
            utils.set_seed(seed)
            if had_mode == 'hadamard':
                K = 1
                had_K = None
            elif had_mode == 'rh':
                had_K = rotation_utils.get_orthogonal_matrix(rank, 'hadamard')
                K = rank
            elif had_mode == 'random':
                had_K = rotation_utils.get_orthogonal_matrix(rank, 'random')
                K = rank
        # Update total rank sum and linear layer count
        total_rank_sum[module_name] += rank
        total_linear_count[module_name] += 1
        logging.info(f"Layer {module_name} uses rank: {rank}")
        
        U = U.view(2, -1, rank)

        Us = [U[0], U[1]]
        Vs = [V]

        if linear.bias is not None:
            biask = linear.bias.data
            biasv = linear1.bias.data
        else:
            biask = None
            biasv = None
        
        if had_rank:
            new_linearK = SVDLinear(Us[0], None, Vs[0], biask, sigma_fuse, had_K, K, had_mode)
            new_linearV = SVDLinear(Us[1], None, Vs[0], biasv, sigma_fuse, had_K, K, had_mode)
        else:
            new_linearK = SVDLinear(Us[0], None, Vs[0], biask, sigma_fuse)
            new_linearV = SVDLinear(Us[1], None, Vs[0], biasv, sigma_fuse)
        _copy_cached_calibration_stats(linear, new_linearK, module_name=module_name)
        _copy_cached_calibration_stats(linear1, new_linearV, module_name="v_proj" if module_name == "k_proj" else module_name)
        new_linearK.to(linear.weight.dtype)
        new_linearV.to(linear.weight.dtype)
        return new_linearK.cpu(), new_linearV.cpu()

    @staticmethod
    def from_linearqkv_with_grad(
        linear: nn.Linear,
        linear1: nn.Linear,
        linear2: nn.Linear,
        param_ratio: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1.,
        sigma_fuse="UV",
        rank_align=1.,
        had_rank=False,
        had_mode='hadamard',
        singular_indices=None,
        seed=0,
        module_name='k_proj',
        is_per_head_svd=False,
        is_q_headnum=False,
    ):

        device = utils.get_dev()
        svd_info = linear1.qkv_svd_info

        # Get SVD information
        U = svd_info['U'].to(device)
        S = svd_info['S'].to(device)
        V = svd_info['V'].to(device)


        # Use pre-calculated importance sorting results
        if singular_indices is not None:
            if is_per_head_svd:
                # U: n, c, c
                # S: n, c
                # V:  n, C_in, c 
                U_selected = []
                S_selected = []
                V_selected = []
                rank_split = []
                N_heads = U.shape[0]
                for head_idx in range(N_heads):
                    head_indices = singular_indices.get(head_idx, []) if isinstance(singular_indices, dict) else singular_indices[head_idx]
                    if len(head_indices) == 0:
                        head_indices = [0]
                        logging.warning(f"No singular index selected for head {head_idx}, fallback to top-1.")
                    U_selected.append(U[head_idx, :, head_indices])
                    S_selected.append(S[head_idx, head_indices])
                    V_selected.append(V[head_idx, :, head_indices]) # fuse in from linear?
                    rank = len(head_indices)
                    logging.info(f"Layer uses rank: {rank} on head {head_idx}")
                    total_rank_sum[module_name] += rank
                    rank_split.append(rank)
                total_linear_count[module_name] += 1
                if had_rank:
                    logging.info("Warning: head wise svd is not supported for had_rank yet")
            else:
                logging.info(f"Using {len(singular_indices)} pre-selected singular values")

                # Select important singular values and corresponding vectors
                U_selected = U[:, singular_indices]
                S_selected = S[singular_indices]
                V_selected = V[:, singular_indices]
                
                # Update rank to the number of selected singular values
                rank = len(singular_indices)

                if had_rank:
                    utils.set_seed(seed)
                    if had_mode == 'hadamard':
                        K = 1
                        had_K = None
                    elif had_mode == 'rh':
                        had_K = rotation_utils.get_orthogonal_matrix(rank, 'hadamard')
                        K = rank    
                    elif had_mode == 'random':
                        had_K = rotation_utils.get_orthogonal_matrix(rank, 'random')
                        K = rank
                
                # Update total rank sum and linear layer count
                total_rank_sum[module_name] += rank
                total_linear_count[module_name] += 1

                logging.info(f"Layer uses rank: {rank}")
            
            # Process activation-aware scaling
            if hasattr(linear1, 'scaling_diag_matrixS') and linear1.scaling_diag_matrixS is not None:
                #  SVD-LLM
                linear1.scaling_diag_matrix = linear1.scaling_diag_matrixS
                
            if act_aware and linear1.scaling_diag_matrix is not None:
                scaling_diag_matrix = linear1.scaling_diag_matrix.to(device)
                if scaling_diag_matrix.ndim == 1:
                    # ASVD
                    # One-dimensional vector, representing diagonal matrix diagonal elements
                    scaling_diag_matrix = scaling_diag_matrix**alpha
                    scaling_diag_matrix += 1e-6  # avoid zero division
                    if is_per_head_svd:
                        V_selected  = [v_select / scaling_diag_matrix.view(-1, 1) for v_select in V_selected]
                    else:
                        V_selected = V_selected / scaling_diag_matrix.view(-1, 1)
                elif scaling_diag_matrix.ndim == 2:
                    # SVD-LLM
                    # Two-dimensional matrix, complete scaling matrix, need to right multiply inverse matrix
                    try:
                        scaling_diag_matrix_inv = torch.linalg.inv(scaling_diag_matrix).to(torch.float32)
                    except RuntimeError as e:
                        logging.info("Warning: scaling_diag_matrix is not full rank, adding epsilon for stability.")
                        eps = 1e-6
                        scaling_diag_matrix += eps * torch.eye(scaling_diag_matrix.shape[0], device=device)
                        scaling_diag_matrix_inv = torch.linalg.inv(scaling_diag_matrix).to(torch.float32)
                        
                    if is_per_head_svd:
                        V_selected = [scaling_diag_matrix_inv.T @ v_select for v_select in V_selected]
                    else:
                        V_selected = scaling_diag_matrix_inv.T @ V_selected

            # Split U into three parts, corresponding to q, k, v
            if is_per_head_svd:
                if linear.out_features == linear1.out_features:
                    Us = [[], [], []]
                    for i in range(len(U_selected)):
                        rank_ = U_selected[i].shape[-1] # # n* [3c, rank] (c*n = C_out)
                        u_select = U_selected[i].view(3, -1, rank_)
                        Us[0].append(u_select[0])
                        Us[1].append(u_select[1])
                        Us[2].append(u_select[2])
                else:
                    U_selected = torch.split(U_selected, [linear.out_features//N_heads, linear1.out_features//N_heads, linear2.out_features//N_heads], dim=1)
                    Us = [U_selected[0], U_selected[1], U_selected[2]]
                Ss = [S_selected]
                Vs = [V_selected]
            else:
                if linear.out_features == linear1.out_features:
                    U_selected = U_selected.view(3, -1, rank)
                else: # GQA specifically
                    U_selected = torch.split(U_selected, [linear.out_features, linear1.out_features, linear2.out_features], dim=0)
                Us = [U_selected[0], U_selected[1], U_selected[2]]
                Ss = [S_selected]
                Vs = [V_selected]

            # Process bias
            if linear.bias is not None:
                biasq = linear.bias.data
                biask = linear1.bias.data
                biasv = linear2.bias.data
            else:
                biasq = None
                biask = None
                biasv = None

            # Check NaN or Inf
            if not is_per_head_svd:
                for S in Ss:
                    if (S != S).any():
                        logging.info("nan in S")
                        return (
                            nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                            nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                            nn.Linear(linear2.in_features, linear2.out_features).to(linear2.weight.dtype).to(linear2.weight.device)
                        )
                for U in Us:
                    if (U != U).any():
                        logging.info("nan in U")
                        return (
                            nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                            nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                            nn.Linear(linear2.in_features, linear2.out_features).to(linear2.weight.dtype).to(linear2.weight.device)
                        )
                for V in Vs:
                    if (V != V).any():
                        logging.info("nan in V")
                        return (
                            nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                            nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                            nn.Linear(linear2.in_features, linear2.out_features).to(linear2.weight.dtype).to(linear2.weight.device)
                        )
                assert len(Us)/3 == len(Ss) == len(Vs) == 1

            # Create new linear layer
            if had_rank:
                if is_per_head_svd:
                    logging.info("Warning: head wise svd is not supported for had_rank yet")
                new_linearQ = SVDLinear(Us[0], Ss[0], Vs[0], biasq, sigma_fuse, had_K, K, had_mode)
                new_linearK = SVDLinear(Us[1], Ss[0], Vs[0], biask, sigma_fuse, had_K, K, had_mode)
                new_linearV = SVDLinear(Us[2], Ss[0], Vs[0], biasv, sigma_fuse, had_K, K, had_mode)
            else:
                if is_per_head_svd:
                    new_linearQ = SVDLinear(Us[0], Ss[0], Vs[0], biasq, sigma_fuse, is_per_head_svd=is_per_head_svd, rank_split=rank_split)
                    new_linearK = SVDLinear(Us[1], Ss[0], Vs[0], biask, sigma_fuse, is_per_head_svd=is_per_head_svd, rank_split=rank_split)
                    new_linearV = SVDLinear(Us[2], Ss[0], Vs[0], biasv, sigma_fuse, is_per_head_svd=is_per_head_svd, rank_split=rank_split)
                else:
                    new_linearQ = SVDLinear(Us[0], Ss[0], Vs[0], biasq, sigma_fuse)
                    new_linearK = SVDLinear(Us[1], Ss[0], Vs[0], biask, sigma_fuse)
                    new_linearV = SVDLinear(Us[2], Ss[0], Vs[0], biasv, sigma_fuse)
            _copy_cached_calibration_stats(linear, new_linearQ, module_name="q_proj")
            _copy_cached_calibration_stats(linear1, new_linearK, module_name="k_proj")
            _copy_cached_calibration_stats(linear2, new_linearV, module_name="v_proj")
            new_linearQ.to(linear.weight.dtype)
            new_linearK.to(linear.weight.dtype)
            new_linearV.to(linear.weight.dtype)
            return new_linearQ.cpu(), new_linearK.cpu(), new_linearV.cpu()


    @staticmethod
    def from_linear_with_grad(
        linear: nn.Linear,
        param_ratio: float,
        act_aware=False,
        loss_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1.,
        sigma_fuse="UV",
        rank_align=1.,
        had_rank=False,
        had_mode='hadamard',
        singular_indices=None,
        seed=0,
        module_name='o_proj',
        is_ma_hack=False,
        topk=1,
        is_per_head_svd=False,
        is_q_headnum=False,
    ):
        
        device = utils.get_dev()
        svd_info = linear.svd_info

        # Get SVD information
        U = svd_info['U'].to(device)
        S = svd_info['S'].to(device)
        V = svd_info['V'].to(device)

        # Use pre-calculated importance sorting results
        if singular_indices is not None:
            if is_per_head_svd:
                # U: n, c, c
                # S: n, c
                # V:  n, C_in, c 
                U_selected = []
                S_selected = []
                V_selected = []
                rank_split = []
                N_heads = U.shape[0]
                for head_idx in range(N_heads):
                    head_indices = singular_indices.get(head_idx, []) if isinstance(singular_indices, dict) else singular_indices[head_idx]
                    if len(head_indices) == 0:
                        head_indices = [0]
                        logging.warning(f"No singular index selected for head {head_idx}, fallback to top-1.")
                    U_selected.append(U[head_idx, :, head_indices])
                    S_selected.append(S[head_idx, head_indices])
                    V_selected.append(V[head_idx, :, head_indices])
                    rank = len(head_indices)
                    logging.info(f"Layer uses rank: {rank} on head {head_idx}")
                    total_rank_sum[module_name] += rank
                    rank_split.append(rank)
                total_linear_count[module_name] += 1

                if had_rank:
                    logging.info("Warning: head wise svd is not supported for had_rank yet")
            else:
                logging.info(f"Using {len(singular_indices)} pre-selected singular values")

                # Select important singular values and corresponding vectors
                U_selected = U[:, singular_indices]
                S_selected = S[singular_indices]
                V_selected = V[:, singular_indices]

                # Update rank to the number of selected singular values
                rank = len(singular_indices)

                if had_rank:
                    utils.set_seed(seed)
                    if had_mode == 'hadamard':
                        K = 1
                        had_K = None
                    elif had_mode == 'rh':
                        had_K = rotation_utils.get_orthogonal_matrix(rank, 'hadamard')
                        K = rank    
                    elif had_mode == 'random':
                        had_K = rotation_utils.get_orthogonal_matrix(rank, 'random')
                        K = rank
                    
                # Update total rank sum and linear layer count
                total_rank_sum[module_name] += rank
                total_linear_count[module_name] += 1

                logging.info(f"Layer uses rank: {rank}")

            # Process activation-aware scaling
            scaling_diag_matrix = None
            if act_aware:
                try:
                    scaling_diag_matrix = linear.scaling_diag_matrix
                except:
                    scaling_diag_matrix = linear.scaling_diag_matrixS
            if act_aware and scaling_diag_matrix is not None:
                scaling_diag_matrix = scaling_diag_matrix.to(device)
                if scaling_diag_matrix.ndim == 1:
                    # ASVD
                    # One-dimensional vector, representing diagonal matrix diagonal elements
                    scaling_diag_matrix = scaling_diag_matrix**alpha
                    scaling_diag_matrix += 1e-6  # avoid zero division
                    if is_per_head_svd:
                        V_selected = [v_select / scaling_diag_matrix.view(-1, 1) for v_select in V_selected]
                    else:
                        V_selected = V_selected / scaling_diag_matrix.view(-1, 1)
                elif scaling_diag_matrix.ndim == 2:
                    # SVD-LLM
                    # Two-dimensional matrix, complete scaling matrix, need to right multiply inverse matrix
                    try:
                        scaling_diag_matrix_inv = torch.linalg.inv(scaling_diag_matrix).to(torch.float32)
                    except RuntimeError as e:
                        logging.info("Warning: scaling_diag_matrix is not full rank, adding epsilon for stability.")
                        eps = 1e-6
                        scaling_diag_matrix += eps * torch.eye(scaling_diag_matrix.shape[0], device=device)
                        scaling_diag_matrix_inv = torch.linalg.inv(scaling_diag_matrix).to(torch.float32)
                    
                    if is_per_head_svd:
                        V_selected = [scaling_diag_matrix_inv.T @ v_select for v_select in V_selected]
                    else:
                        V_selected = scaling_diag_matrix_inv.T @ V_selected

            if loss_aware:
                output_grad_scaling_matrix_inv = svd_info['output_grad_scaling_matrix_inv'].to(device)
                if is_per_head_svd:
                    U_selected = [output_grad_scaling_matrix_inv[head_idx].float() @ u_select for head_idx, u_select in enumerate(U_selected)]
                else:
                    U_selected = output_grad_scaling_matrix_inv.float() @ U_selected
                    
            Us = [U_selected]
            Ss = [S_selected]
            Vs = [V_selected]

            # Process bias
            if linear.bias is not None:
                bias = linear.bias.data
            else:
                bias = None

            # Check NaN or Inf
            if not is_per_head_svd:
                for S in Ss:
                    if (S != S).any():
                        logging.info("nan in S")
                        return (
                            nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                        )
                for U in Us:
                    if (U != U).any():
                        logging.info("nan in U")
                        return (
                            nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                        )
                for V in Vs:
                    if (V != V).any():
                        logging.info("nan in V")
                        return (
                            nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                        )
                assert len(Us) == len(Ss) == len(Vs) == 1
            
            # Create new linear layer
            if had_rank:
                if is_per_head_svd:
                    logging.info("Warning: head wise svd is not supported for had_rank yet. Fallback to no had_rank.")
                    new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear if is_ma_hack else None, topk=topk, is_per_head_svd=is_per_head_svd, rank_split=rank_split)
                else:
                    new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse, had_K, K, had_mode, is_ma_hack, linear if is_ma_hack else None, topk)
            else:
                if is_per_head_svd:
                    new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear if is_ma_hack else None, topk=topk, is_per_head_svd=is_per_head_svd, rank_split=rank_split)
                else:
                    new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear if is_ma_hack else None, topk=topk)
            _copy_cached_calibration_stats(linear, new_linear, module_name=module_name)
            new_linear.to(linear.weight.dtype)
            return new_linear.cpu()

    @staticmethod
    def from_linear_with_loss_aware_and_truncate(
        linear: nn.Linear,
        param_ratio: float,
        act_aware=False,
        loss_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1.,
        sigma_fuse="UV",
        rank_align=1.,
        had_rank=False,
        had_mode='hadamard',
        singular_indices=None,
        seed=0,
        module_name='o_proj',
        is_ma_hack=False,
        topk=1,
        is_per_head_svd=False,
        is_q_headnum=False,
    ):
        device = utils.get_dev()
        svd_info = linear.svd_info

        U = svd_info['U'].to(device)
        S = svd_info['S'].to(device)
        V = svd_info['V'].to(device)

        if singular_indices is not None:
            if is_per_head_svd:
                U_selected = []
                S_selected = []
                V_selected = []
                rank_split = []
                N_heads = U.shape[0]
                for head_idx in range(N_heads):
                    head_indices = singular_indices.get(head_idx, []) if isinstance(singular_indices, dict) else singular_indices[head_idx]
                    if len(head_indices) == 0:
                        head_indices = [0]
                        logging.warning(f"No singular index selected for head {head_idx}, fallback to top-1.")
                    U_selected.append(U[head_idx, :, head_indices])
                    S_selected.append(S[head_idx, head_indices])
                    V_selected.append(V[head_idx, :, head_indices])
                    rank = len(head_indices)
                    logging.info(f"Layer uses rank: {rank} on head {head_idx}")
                    total_rank_sum[module_name] += rank
                    rank_split.append(rank)
                total_linear_count[module_name] += 1

                if had_rank:
                    logging.info("Warning: head wise svd is not supported for had_rank yet")
            else:
                logging.info(f"Using {len(singular_indices)} pre-selected singular values")

                U_selected = U[:, singular_indices]
                S_selected = S[singular_indices]
                V_selected = V[:, singular_indices]
                rank = len(singular_indices)

                if had_rank:
                    utils.set_seed(seed)
                    if had_mode == 'hadamard':
                        K = 1
                        had_K = None
                    elif had_mode == 'rh':
                        had_K = rotation_utils.get_orthogonal_matrix(rank, 'hadamard')
                        K = rank
                    elif had_mode == 'random':
                        had_K = rotation_utils.get_orthogonal_matrix(rank, 'random')
                        K = rank

                total_rank_sum[module_name] += rank
                total_linear_count[module_name] += 1
                logging.info(f"Layer uses rank: {rank}")

            # already inverse whitening in loss_aware_utils.compute_whitened_singular_values
            Us = [U_selected]
            Ss = [S_selected]
            Vs = [V_selected]

            if linear.bias is not None:
                bias = linear.bias.data
            else:
                bias = None

            if not is_per_head_svd:
                for S in Ss:
                    if (S != S).any():
                        logging.info("nan in S")
                        return (
                            nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                        )
                for U in Us:
                    if (U != U).any():
                        logging.info("nan in U")
                        return (
                            nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                        )
                for V in Vs:
                    if (V != V).any():
                        logging.info("nan in V")
                        return (
                            nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                        )
                assert len(Us) == len(Ss) == len(Vs) == 1

            if had_rank:
                if is_per_head_svd:
                    logging.info("Warning: head wise svd is not supported for had_rank yet")
                else:
                    new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse, had_K, K, had_mode, is_ma_hack, linear if is_ma_hack else None, topk)
            else:
                if is_per_head_svd:
                    new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear if is_ma_hack else None, topk=topk, is_per_head_svd=is_per_head_svd, rank_split=rank_split)
                else:
                    new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear if is_ma_hack else None, topk=topk)
            _copy_cached_calibration_stats(linear, new_linear, module_name=module_name)
            new_linear.to(linear.weight.dtype)
            return new_linear.cpu()

    @staticmethod
    def from_linearkv_with_grad(
        linear: nn.Linear,
        linear1: nn.Linear,
        param_ratio: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1.,
        sigma_fuse="UV",
        rank_align=1.,
        had_rank=False,
        had_mode='hadamard',
        singular_indices=None,
        seed=0,
        module_name='up_proj',
        is_ma_hack=False,
        topk=1,
    ):
        
        device = utils.get_dev()
        svd_info = linear.svd_info

        # Get SVD information
        U = svd_info['U'].to(device)
        S = svd_info['S'].to(device)
        V = svd_info['V'].to(device)

        # Use pre-calculated importance sorting results
        if singular_indices is not None:
            logging.info(f"Using {len(singular_indices)} pre-selected singular values")

            # Select important singular values and corresponding vectors
            U_selected = U[:, singular_indices]
            S_selected = S[singular_indices]
            V_selected = V[:, singular_indices]

            # Update rank to the number of selected singular values
            rank = len(singular_indices)

            if had_rank:
                utils.set_seed(seed)
                if had_mode == 'hadamard':
                    K = 1
                    had_K = None
                elif had_mode == 'rh':
                    had_K = rotation_utils.get_orthogonal_matrix(rank, 'hadamard')
                    K = rank    
                elif had_mode == 'random':
                    had_K = rotation_utils.get_orthogonal_matrix(rank, 'random')
                    K = rank
            
            # Update total rank sum and linear layer count
            total_rank_sum[module_name] += rank
            total_linear_count[module_name] += 1

            logging.info(f"Layer uses rank: {rank}")

            # Process activation-aware scaling
            if act_aware and linear.scaling_diag_matrix is not None:
                scaling_diag_matrix = linear.scaling_diag_matrix.to(device)
                if scaling_diag_matrix.ndim == 1:
                    # ASVD
                    # One-dimensional vector, representing diagonal matrix diagonal elements
                    scaling_diag_matrix = scaling_diag_matrix**alpha
                    scaling_diag_matrix += 1e-6  # avoid zero division
                    V_selected = V_selected / scaling_diag_matrix.view(-1, 1)
                elif scaling_diag_matrix.ndim == 2:
                    # SVD-LLM
                    # Two-dimensional matrix, complete scaling matrix, need to right multiply inverse matrix
                    try:
                        scaling_diag_matrix_inv = torch.linalg.inv(scaling_diag_matrix).to(torch.float32)
                    except RuntimeError as e:
                        logging.info("Warning: scaling_diag_matrix is not full rank, adding epsilon for stability.")
                        eps = 1e-6
                        scaling_diag_matrix += eps * torch.eye(scaling_diag_matrix.shape[0], device=device)
                        scaling_diag_matrix_inv = torch.linalg.inv(scaling_diag_matrix).to(torch.float32)
                    
                    V_selected = scaling_diag_matrix_inv.T @ V_selected

            # Split U into two parts, corresponding to k and v / up and gated
            U_selected = U_selected.view(2, -1, rank)
            Us = [U_selected[0], U_selected[1]]
            Ss = [S_selected]
            Vs = [V_selected]

            # process bias
            if linear.bias is not None:
                biask = linear.bias.data
                biasv = linear1.bias.data
            else:
                biask = None
                biasv = None

            # Check NaN or Inf
            for S in Ss:
                if (S != S).any():
                    logging.info("nan in S")
                    return (
                        nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                        nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                    )
            for U in Us:
                if (U != U).any():
                    logging.info("nan in U")
                    return (
                        nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                        nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                    )
            for V in Vs:
                if (V != V).any():
                    logging.info("nan in V")
                    return (
                        nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                        nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                    )
            assert len(Us)/2 == len(Ss) == len(Vs) == 1

            # Create new linear layer
            if had_rank:
                new_linearK = SVDLinear(Us[0], Ss[0], Vs[0], biask, sigma_fuse, had_K, K, had_mode)
                new_linearV = SVDLinear(Us[1], Ss[0], Vs[0], biasv, sigma_fuse, had_K, K, had_mode)
            else:
                new_linearK = SVDLinear(Us[0], Ss[0], Vs[0], biask, sigma_fuse)
                new_linearV = SVDLinear(Us[1], Ss[0], Vs[0], biasv, sigma_fuse)
            _copy_cached_calibration_stats(linear, new_linearK, module_name=module_name)
            _copy_cached_calibration_stats(linear1, new_linearV, module_name="v_proj" if module_name == "k_proj" else module_name)
            new_linearK.to(linear.weight.dtype)
            new_linearV.to(linear.weight.dtype)
            return new_linearK.cpu(), new_linearV.cpu()

    def forward(self, inp):
        if self.is_ma_hack and self.original_linear is not None:
            # Compute both original and SVD outputs
            original_output = self.original_linear(inp)
            svd_output = self.BLinear(inp)
            svd_output = self.ALinear(svd_output)
            
            # Get top-k locations and values from original output
            original_abs = original_output.abs()
            topk_locs, topk_vals = self.get_topk_location(original_abs, self.topk)
            
            # Create a copy of SVD output to modify
            result = svd_output.clone()
            
            # Replace SVD output with top-k values from original output at corresponding locations
            for rank in range(abs(self.topk)):
                rank_str = str(rank)
                if rank_str in topk_locs and rank_str in topk_vals:
                    locations = topk_locs[rank_str]  # [bsz, 2] with (n_idx, c_idx)
                    values = topk_vals[rank_str]     # [bsz]
                    
                    # Apply the replacement for each batch item
                    for bsz_idx in range(locations.shape[0]):
                        n_idx = locations[bsz_idx, 0].item()
                        c_idx = locations[bsz_idx, 1].item()
                        orig_val = original_output[bsz_idx, n_idx, c_idx]
                        print(f"replace result: {result[bsz_idx, n_idx, c_idx]} with {orig_val}")
                        # Replace the SVD output at the same location with original value
                        result[bsz_idx, n_idx, c_idx] = orig_val
            
            return result
        else:
            if self.is_per_head_svd:
                # per head SVD forward pass
                y = self.BLinear(inp)
                if self.rank_split is not None:


                    # y: (B, T, Rsum)
                    B, T, Rsum = y.shape
                    H = self.head_num
                    ranks = list(self.rank_split)
                    assert len(ranks) == H, f"len(rank_split)={len(ranks)} != head_num={H}"

                    # Sanity: weight ranks match
                    ranks_w = [w.shape[1] for w in self.ALinear]  # each w: (cout, r_h)
                    if ranks_w != ranks:
                        raise ValueError(f"Per-head rank mismatch: weight {ranks_w} vs split {ranks}")

                    Rmax = max(ranks)
                    cout = self.ALinear[0].shape[0]

                    # Split y into heads and pad to (B, T, H, Rmax)
                    y_split = torch.split(y, ranks, dim=-1)             # list of (B, T, r_h)
                    y_pad = y.new_zeros(B, T, H, Rmax)                  # (B,T,H,Rmax)
                    for h, (yh, r) in enumerate(zip(y_split, ranks)):
                        y_pad[:, :, h, :r] = yh

                    # Pad weights to (H, cout, Rmax)
                    w_pad = y.new_zeros(H, cout, Rmax)                  # same device/dtype as y
                    for h, (wh, r) in enumerate(zip(self.ALinear, ranks)):
                        # wh expected shape: (cout, r)
                        w_pad[h, :, :r] = wh

                    # Batched per-head matmul:
                    # (B,T,H,Rmax) x (H,cout,Rmax) -> (B,T,H,cout)
                    y = torch.einsum('bthr,hcr->bthc', y_pad, w_pad)

                    # Flatten heads: (B, T, H*cout)
                    y = y.reshape(B, T, H * cout)

                else:
                    if inp.dim() >= 1 and inp.shape[0] != 1 and not getattr(self, "_warned_per_head_batch", False):
                        logging.warning(
                            "Per-head SVD forward without rank_split currently assumes batch size 1; "
                            "observed input shape %s. Results for batch>1 may be incorrect.",
                            tuple(inp.shape),
                        )
                        self._warned_per_head_batch = True
                    y = y.view(-1, self.head_num, self.head_rank).transpose(0, 1) # (head_num, token, head_rank)
                    y = torch.matmul(y, self.ALinear.transpose(1, 2)) # (head_num, token, head_rank) @ (head_num, c, head_rank).T = (head_num, token, c) -> 
                    y = y.transpose(0, 1).flatten(1).unsqueeze(0)
                if self.bias is not None:
                    bias_view = (1,) * (y.dim() - 1) + (-1,)
                    y = y + self.bias.to(dtype=y.dtype).view(*bias_view)
                return y
            else:
                # Standard SVD forward pass
                out_dtype = inp.dtype if inp.is_floating_point() else None
                y = self.BLinear(_match_module_weight_dtype(self.BLinear, inp))
                if out_dtype is not None and y.dtype != out_dtype:
                    y = y.to(out_dtype)
                y = self.ALinear(_match_module_weight_dtype(self.ALinear, y))
                if out_dtype is not None and y.dtype != out_dtype:
                    y = y.to(out_dtype)
                return y

def rsetattr(obj, attr, value):
    """ Recursively set an attribute given a dotted path """
    pre, _, post = attr.rpartition('.')
    if pre:
        obj = rgetattr(obj, pre)  # Get the nested object first
    setattr(obj, post, value)

def rgetattr(obj, attr):
    """ Recursively get an attribute given a dotted path """
    for part in attr.split('.'):
        obj = getattr(obj, part)
    return obj


def _unwrap_actquant_linear(module):
    if isinstance(module, quant_utils.ActQuantWrapper):
        return module.module
    return module


def _find_svd_candidate_layers(module):
    return quant_utils.find_qlayers(
        module, layers=[torch.nn.Linear, quant_utils.ActQuantWrapper]
    )


# @torch.inference_mode()
def svd_lm_setup(model, args, tokenizer=None, image_processor=None):
    global total_rank_sum, total_linear_count
    if reasoning_calib_utils.enabled(args):
        reasoning_calib_utils.cache_tag(
            args,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
        )
    # Reset counters
    total_rank_sum = {
        'q_proj': 0,
        'v_proj': 0,
        'k_proj': 0,
        'o_proj': 0,
        'up_proj': 0,
        'gate_proj': 0,
        'down_proj': 0
    }
    total_linear_count = {
        'q_proj': 0,
        'v_proj': 0,
        'k_proj': 0,
        'o_proj': 0,
        'up_proj': 0,
        'gate_proj': 0,
        'down_proj': 0
    }
    
    if args.act_aware or args.fisher_info or args.grad_info or getattr(args, 'latent_smooth', False) or args.loss_aware:
        # Load calibration dataset
        logging.info(f"Loading calibration dataset: {args.cal_dataset}")
        calib_loader = data_utils.get_loaders(
            args.cal_dataset, 
            nsamples=args.nsamples,
            seed=args.seed, 
            model=args.model,
            seqlen=model.seqlen, 
            eval_mode=False,
            hf_token=args.hf_token 
        )
        if 'wiki' in args.cal_dataset:
            dataloader = calib_loader
        else:
            dataloader, _ = calib_loader
        # Apply text-image misalignment if requested
        if hasattr(args, 'misalign_text_image') and args.misalign_text_image:
            logging.info("Applying text-image misalignment to calibration dataloader")
            logging.info(f"Original dataloader type: {type(dataloader)}")
            if isinstance(dataloader, list):
                logging.info(f"Original dataloader length: {len(dataloader)}")
            dataloader = grad_info_utils.create_misaligned_dataloader(dataloader, seed=args.seed if hasattr(args, 'seed') else None)
            logging.info("Text-image misalignment applied successfully")
        else:
            logging.info("No text-image misalignment requested")
    # Decide whether to calibrate based on args
    if args.act_aware or args.loss_aware:
        temp_svd_modules = args.svd_modules
        if 'all' in args.svd_modules:
            args.svd_modules = 'all'
        # Qwen3-VL's MRoPE/deepstack state is not reproduced correctly by the
        # low-resource layer-by-layer replay path, so keep cholesky but use
        # normal full-forward calibration for that family.
        use_low_resource_cholesky = args.calib_method == 'cholesky'
        if use_low_resource_cholesky and _is_qwen3_vl_model(args, model, tokenizer):
            logging.info(
                "Qwen3-VL detected: using full-forward cholesky calibration instead of low-resource replay."
            )
            use_low_resource_cholesky = False
        if use_low_resource_cholesky and molmo_utils.is_molmo_model(args=args, model=model, tokenizer=tokenizer):
            logging.info(
                "Molmo detected: using full-forward cholesky calibration instead of low-resource replay."
            )
            use_low_resource_cholesky = False
        if use_low_resource_cholesky and reasoning_calib_utils.enabled(args):
            logging.info(
                "Reasoning calibration enabled: using full-forward cholesky calibration instead of low-resource replay."
            )
            use_low_resource_cholesky = False
        # Perform calibration - directly use cache function in act_aware_utils
        if use_low_resource_cholesky: #  calib_input_distributionlowresources
            logging.info('using low resource version of SVDLLM')
            act_aware_utils.calib_input_distributionlowresources(
                model=model, 
                dataloader=dataloader,
                tokenizer=tokenizer, 
                image_processor=image_processor, 
                args=args, 
                method=args.calib_method, 
                use_cache=args.use_cache,  # default enable cache
                cache_file=None
            )
        else:
            act_aware_utils.calib_input_distribution(
                model=model, 
                dataloader=dataloader,
                tokenizer=tokenizer, 
                image_processor=image_processor, 
                args=args, 
                method=args.calib_method, 
                use_cache=args.use_cache,  # default enable cache
                cache_file=None
            )
        if 'all' in temp_svd_modules:
            args.svd_modules = temp_svd_modules

    if args.loss_aware:
        if 'qkv' in args.svd_modules:
            loss_aware_utils.calib_grad_distribution(
                model=model,
                dataloader=dataloader,
                tokenizer=tokenizer,
                image_processor=image_processor,
                args=args,
                use_cache=args.use_cache,
                cache_file=None
            )
        else:
            loss_aware_utils.calib_grad_distribution_low_resources(
                model=model,
                dataloader=dataloader,
                tokenizer=tokenizer,
                image_processor=image_processor,
                args=args,
                use_cache=args.use_cache,
                cache_file=None
            )

    if args.loss_aware_and_truncate:
        args.loss_aware = True
        model_type = model_utils.get_model_type(model)
        utils.cleanup_memory()
        layers = model_utils.get_transformer_layers(model, model_type=model_type)

        loss_aware_utils.prepare_svd(model, args)
        top_indices, top_scores, layer_indices_dict = loss_aware_utils.svd_with_loss_aware(layers, args)
        if hasattr(args, 'truncate_smooth_max_factor') and args.truncate_smooth_max_factor is not None:
            smooth_top_indices, smooth_top_scores, smooth_layer_indices_dict = smooth_truncate_selection(
                top_indices=top_indices,
                top_scores=top_scores,
                num_layers=len(layers),
                keys=list(top_indices.keys()),
                max_factor=float(args.truncate_smooth_max_factor),
                is_per_head_svd=getattr(args, 'is_per_head_svd', False),
            )
            layer_indices_dict = smooth_layer_indices_dict
        rank, world_size = get_rank_and_world_size()

        for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="LM SVD loss_aware_and_truncate")):
            full = _find_svd_candidate_layers(layers[idx])
            for name, module in full.items():
                if args.qkv_fuse:
                    raise NotImplementedError("QKV fuse is not supported for loss_aware_and_truncate")
                elif args.kv_fuse:
                    raise NotImplementedError("KV fuse is not supported for loss_aware_and_truncate")
                # elif args.inp_fuse:
                #     raise NotImplementedError("Inp fuse is not supported for loss_aware_and_truncate")
                else:
                    if ('q_proj' in name or 'k_proj' in name or 'v_proj' in name) and args.svd_modules in ['qkv', 'attn', 'all']:
                        q_dense = _unwrap_actquant_linear(layer.self_attn.q_proj)
                        k_dense = _unwrap_actquant_linear(layer.self_attn.k_proj)
                        v_dense = _unwrap_actquant_linear(layer.self_attn.v_proj)
                        if (
                            q_dense is not None
                            and k_dense is not None
                            and v_dense is not None
                            and hasattr(k_dense, "scaling_diag_matrixS")
                            and k_dense.scaling_diag_matrixS is not None
                        ):
                            q_dense.scaling_diag_matrixS = k_dense.scaling_diag_matrixS
                            v_dense.scaling_diag_matrixS = k_dense.scaling_diag_matrixS
                        if 'q_proj' in name:
                            proj_name = 'q_proj'
                            proj = layer.self_attn.q_proj
                        elif 'k_proj' in name:
                            proj_name = 'k_proj'
                            proj = layer.self_attn.k_proj
                        elif 'v_proj' in name:
                            proj_name = 'v_proj'
                            proj = layer.self_attn.v_proj

                        if idx not in layer_indices_dict or proj_name not in layer_indices_dict[idx]:
                            logging.info(f"Layer {idx} {proj_name} is not in layer_indices_dict, skipping")
                            continue

                        if rank == 0:
                            logging.info(f"Layer {idx} starting {proj_name} SVD (loss_aware_and_truncate), using {len(layer_indices_dict[idx][proj_name])} singular values")
                        try:
                            new_linear = SVDLinear.from_linear_with_loss_aware_and_truncate(
                                _unwrap_actquant_linear(proj),
                                param_ratio=args.rank_ratio,
                                alpha=args.act_alpha,
                                act_aware=args.act_aware,
                                loss_aware=True,
                                rank_align=1.,
                                sigma_fuse=args.svd_mode,
                                had_rank=args.had_rank,
                                had_mode='random',
                                singular_indices=layer_indices_dict[idx][proj_name],
                                seed=args.seed,
                                module_name=proj_name,
                                is_per_head_svd=args.is_per_head_svd,
                                is_q_headnum=args.is_q_headnum,
                            )
                            rsetattr(layers[idx], f'self_attn.{proj_name}', new_linear)
                            if rank == 0:
                                logging.info(f"Layer {idx} {proj_name} SVD completed")
                        except Exception as e:
                            logging.info(f"Layer {idx} {proj_name} SVD failed: {e}")
                            import traceback
                            logging.error(traceback.format_exc())

                if 'o_proj' in name and args.svd_modules in ['attn', 'all', 'o']:
                    if idx not in layer_indices_dict or 'o_proj' not in layer_indices_dict[idx]:
                        logging.info(f"Layer {idx} o_proj is not in layer_indices_dict, skipping")
                        continue
                    if rank == 0:
                        logging.info(f"Layer {idx} starting o_proj SVD (loss_aware_and_truncate), using {len(layer_indices_dict[idx]['o_proj'])} singular values")
                    try:
                        olinear = SVDLinear.from_linear_with_loss_aware_and_truncate(
                            _unwrap_actquant_linear(layer.self_attn.o_proj),
                            param_ratio=args.rank_ratio,
                            alpha=args.act_alpha,
                            act_aware=args.act_aware,
                            loss_aware=True,
                            rank_align=1.,
                            sigma_fuse=args.svd_mode,
                            had_rank=args.had_rank,
                            had_mode='random',
                            singular_indices=layer_indices_dict[idx]['o_proj'],
                            seed=args.seed,
                            module_name='o_proj',
                            is_ma_hack=False,
                            topk=1,
                        )
                        rsetattr(layers[idx], 'self_attn.o_proj', olinear)
                        if rank == 0:
                            logging.info(f"Layer {idx} o_proj fusion SVD completed")
                    except Exception as e:
                        logging.info(f"Layer {idx} o_proj fusion SVD failed: {e}")
                        import traceback
                        logging.error(traceback.format_exc())

                if 'mlp.up_proj' in name and args.svd_modules in ['mlp', 'all', 'gaup']:
                    if idx not in layer_indices_dict or 'up_proj' not in layer_indices_dict[idx]:
                        logging.info(f"Layer {idx} mlp.up_proj is not in layer_indices_dict, skipping")
                        continue
                    if rank == 0:
                        logging.info(f"Layer {idx} starting mlp.up_proj SVD (loss_aware_and_truncate), using {len(layer_indices_dict[idx]['up_proj'])} singular values")
                    try:
                        uplinear = SVDLinear.from_linear_with_loss_aware_and_truncate(
                            layer.mlp.up_proj,
                            param_ratio=args.rank_ratio,
                            alpha=args.act_alpha,
                            act_aware=args.act_aware,
                            loss_aware=True,
                            rank_align=1.,
                            sigma_fuse=args.svd_mode,
                            had_rank=args.had_rank,
                            had_mode='random',
                            singular_indices=layer_indices_dict[idx]['up_proj'],
                            seed=args.seed,
                            module_name='up_proj',
                        )
                        rsetattr(layers[idx], 'mlp.up_proj', uplinear)
                        if rank == 0:
                            logging.info(f"Layer {idx} mlp.up_proj fusion SVD completed")
                    except Exception as e:
                        logging.info(f"Layer {idx} mlp.up_proj SVD failed: {e}")
                        import traceback
                        logging.error(traceback.format_exc())

                if 'mlp.gate_proj' in name and args.svd_modules in ['mlp', 'all', 'gaup']:
                    if idx not in layer_indices_dict or 'gate_proj' not in layer_indices_dict[idx]:
                        continue
                    try:
                        gatedlinear = SVDLinear.from_linear_with_loss_aware_and_truncate(
                            layer.mlp.gate_proj,
                            param_ratio=args.rank_ratio,
                            alpha=args.act_alpha,
                            act_aware=args.act_aware,
                            loss_aware=True,
                            rank_align=1.,
                            sigma_fuse=args.svd_mode,
                            had_rank=args.had_rank,
                            had_mode='random',
                            singular_indices=layer_indices_dict[idx]['gate_proj'],
                            seed=args.seed,
                            module_name='gate_proj',
                        )
                        rsetattr(layers[idx], 'mlp.gate_proj', gatedlinear)
                        if rank == 0:
                            logging.info(f"Layer {idx} mlp.gate_proj fusion SVD completed")
                    except Exception as e:
                        logging.info(f"Layer {idx} mlp.gate_proj SVD failed: {e}")
                        import traceback
                        logging.error(traceback.format_exc())

                if 'mlp.down_proj' in name and args.svd_modules in ['mlp', 'all', 'down']:
                    if idx not in layer_indices_dict or 'down_proj' not in layer_indices_dict[idx]:
                        logging.info(f"Layer {idx} mlp.down_proj is not in layer_indices_dict, skipping")
                        continue
                    if rank == 0:
                        logging.info(f"Layer {idx} starting mlp.down_proj SVD (loss_aware_and_truncate), using {len(layer_indices_dict[idx]['down_proj'])} singular values")
                    try:
                        downlinear = SVDLinear.from_linear_with_loss_aware_and_truncate(
                            _unwrap_actquant_linear(layer.mlp.down_proj),
                            param_ratio=args.rank_ratio,
                            alpha=args.act_alpha,
                            act_aware=args.act_aware,
                            loss_aware=True,
                            rank_align=1.,
                            sigma_fuse=args.svd_mode,
                            had_rank=args.had_rank,
                            had_mode='random',
                            singular_indices=layer_indices_dict[idx]['down_proj'],
                            seed=args.seed,
                            module_name='down_proj',
                        )
                        rsetattr(layers[idx], 'mlp.down_proj', downlinear)
                        if rank == 0:
                            logging.info(f"Layer {idx} mlp.down_proj fusion SVD completed")
                    except Exception as e:
                        logging.info(f"Layer {idx} mlp.down_proj SVD failed: {e}")
                        import traceback
                        logging.error(traceback.format_exc())

        utils.cleanup_memory()
        return

    if args.fisher_info:
        # LSVD release: Fisher-information calibration (utils/fisher_info_utils.py) is not included.
        raise NotImplementedError("--fisher_info is not included in the LSVD release.")
    
    if args.grad_info:

        if args.sw_analyze:
            from . import super_weight_utils
            super_weight_utils.SW_forward_model(model, dataloader, tokenizer, image_processor, args)
            return 
        # Perform gradient information calculation - calib_grad_info includes prepare_qkv_svd call
        if args.svd_lm_localft:
            layers = model_utils.get_transformer_layers(model, model_type=model_utils.get_model_type(model))

            # A loaded quantized checkpoint still needs the fitted SVD module
            # structure to be reconstructed before load_state_dict(). Keep this
            # cache independent of the activation/gradient cache so evaluation
            # can reuse the ScienceQA fit without rerunning local FT.
            default_svd_info_dir = args.cache_file if args.cache_file is not None else args.save_path
            svd_info_dir = getattr(args, 'svd_info_cache_dir', None) or default_svd_info_dir
            svd_info_name = (
                'qkv_svd_ft_info_rank_allocate.pth'
                if args.is_rank_allocate_ft
                else 'qkv_svd_ft_info.pth'
            )
            svd_info_file = os.path.join(svd_info_dir, svd_info_name)
            has_precomputed_svd_info = os.path.isfile(svd_info_file) and os.path.getsize(svd_info_file) > 0
            temp_svd_modules = args.svd_modules

            if args.is_rank_allocate_ft and not has_precomputed_svd_info:
                temp_svd_modules = args.svd_modules
                if args.svd_modules == 'all_sep_rank_allocate_qkvonly':
                    args.svd_modules = 'qkv'
                if 'all_' in args.svd_modules:
                    args.svd_modules = 'all' # rank allocation computed for all
                grad_info_utils.calib_grad_info(
                    model=model,
                    dataloader=dataloader,
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    args=args,
                    use_cache=args.use_cache,  # default enable cache
                    cache_file=None
                )
                _, _, layer_indices_dict = grad_info_utils.svd_qkv_with_grad_info(layers, args, use_cache=args.use_cache) # top_indices: (layer_idx, singular_value_idx)
                # then use top_indices to select each layer's rank etc.
                args.svd_modules = temp_svd_modules
            else:
                layer_indices_dict = None

            from . import local_ft_grad_utils
            if not has_precomputed_svd_info:
                utils.cleanup_memory()
                # for loop here to module wise ft?
                if args.svd_modules == 'all_sep_fine':
                    temp_svd_modules = 'all_sep_fine'
                    args.is_stage = True
                    args.svd_modules = 'qkv'
                    local_ft_grad_utils.calib_localft_grad_info(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, temp_svd_modules, layer_indices_dict)
                    utils.cleanup_memory()
                    local_ft_grad_utils.local_ft_fused_svd(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, layer_indices_dict)
                    args.svd_modules = 'o'
                    local_ft_grad_utils.calib_localft_grad_info(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, temp_svd_modules)
                    utils.cleanup_memory()
                    local_ft_grad_utils.local_ft_fused_svd(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, layer_indices_dict)
                    args.svd_modules = 'gaup'
                    local_ft_grad_utils.calib_localft_grad_info(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, temp_svd_modules)
                    utils.cleanup_memory()
                    local_ft_grad_utils.local_ft_fused_svd(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, layer_indices_dict)
                    args.svd_modules = 'down'
                    local_ft_grad_utils.calib_localft_grad_info(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, temp_svd_modules)
                    utils.cleanup_memory()
                    local_ft_grad_utils.local_ft_fused_svd(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, layer_indices_dict)
                    args.svd_modules = 'all_sep_fine'
                elif args.svd_modules == 'all_sep':
                    temp_svd_modules = 'all_sep'
                    args.is_stage = True
                    args.svd_modules = 'attn'
                    local_ft_grad_utils.calib_localft_grad_info(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, temp_svd_modules)
                    utils.cleanup_memory()
                    local_ft_grad_utils.local_ft_fused_svd(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, layer_indices_dict)
                    args.svd_modules = 'mlp'
                    local_ft_grad_utils.calib_localft_grad_info(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, temp_svd_modules)
                    utils.cleanup_memory()
                    local_ft_grad_utils.local_ft_fused_svd(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, layer_indices_dict)
                    args.svd_modules = 'all_sep'
                elif args.svd_modules == 'all_sep_fine_wo_down':
                    temp_svd_modules = 'all_sep_fine_wo_down'
                    args.is_stage = True
                    args.svd_modules = 'qkv'
                    local_ft_grad_utils.calib_localft_grad_info(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, temp_svd_modules)
                    utils.cleanup_memory()
                    local_ft_grad_utils.local_ft_fused_svd(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, layer_indices_dict)
                    args.svd_modules = 'o'
                    local_ft_grad_utils.calib_localft_grad_info(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, temp_svd_modules)
                    utils.cleanup_memory()
                    local_ft_grad_utils.local_ft_fused_svd(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, layer_indices_dict)
                    args.svd_modules = 'gaup'
                    local_ft_grad_utils.calib_localft_grad_info(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, temp_svd_modules)
                    utils.cleanup_memory()
                    local_ft_grad_utils.local_ft_fused_svd(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, layer_indices_dict)
                    args.svd_modules = 'all_sep_fine_wo_down'
                elif args.svd_modules == 'all_sep_rank_allocate_qkvonly':
                    temp_svd_modules = 'all_sep_rank_allocate_qkvonly'
                    args.svd_modules = 'qkv'
                    args.is_stage = True
                    # why is this a difference? compared with none-stage?
                    local_ft_grad_utils.calib_localft_grad_info(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, temp_svd_modules, layer_indices_dict)
                    utils.cleanup_memory()
                    local_ft_grad_utils.local_ft_fused_svd(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, layer_indices_dict)
                    layer_indices_dict = None
                    args.svd_modules = 'o'
                    local_ft_grad_utils.calib_localft_grad_info(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, temp_svd_modules)
                    utils.cleanup_memory()
                    local_ft_grad_utils.local_ft_fused_svd(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, layer_indices_dict)
                    args.svd_modules = 'gaup'
                    local_ft_grad_utils.calib_localft_grad_info(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, temp_svd_modules)
                    utils.cleanup_memory()
                    local_ft_grad_utils.local_ft_fused_svd(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, layer_indices_dict)
                    args.svd_modules = 'down'
                    local_ft_grad_utils.calib_localft_grad_info(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, temp_svd_modules)
                    utils.cleanup_memory()
                    local_ft_grad_utils.local_ft_fused_svd(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, layer_indices_dict)
                    args.svd_modules = 'all_sep_rank_allocate_qkvonly'
                else:
                    args.is_stage = False # this is to handle qat finetune not all svd 
                    local_ft_grad_utils.calib_localft_grad_info(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, temp_svd_modules, layer_indices_dict)
                    local_ft_grad_utils.local_ft_fused_svd(model, dataloader, tokenizer, image_processor, args, args.use_cache, None, layer_indices_dict)
                utils.cleanup_memory()
                local_ft_grad_utils.save_svd_info(model, args)
            else:
                logging.info(
                    "Reconstructing local-FT SVD modules from precomputed fit cache: %s",
                    svd_info_file,
                )
                local_ft_grad_utils.load_svd_info(model, args, load_path=svd_info_dir)
            ## add from UV to SVDlinear
            
            for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="LM SVD converting localft results ")):
                full = _find_svd_candidate_layers(layers[idx])
                for name, module in full.items():
                    if 'k_proj' in name and args.svd_modules in ['qkv', 'attn', 'all', 'all_sep_fine', 'all_sep_fine_wo_down', 'all_sep','all_sep_rank_allocate_qkvonly']:
                        if args.qkv_fuse:
                            logging.info("Warning: qkv_fuse is not supported for localft yet")
                        elif args.kv_fuse:
                            logging.info("Warning: kv_fuse is not supported for localft yet")
                        else:
                            if not hasattr(layer.self_attn.q_proj, 'svd_info'):
                                logging.info(f"Layer {idx} Q has no pre-calculated SVD information, skipping")
                                continue
                            if not hasattr(layer.self_attn.k_proj, 'svd_info'):
                                logging.info(f"Layer {idx} K has no pre-calculated SVD information, skipping")
                                continue
                            if not hasattr(layer.self_attn.v_proj, 'svd_info'):
                                logging.info(f"Layer {idx} V has no pre-calculated SVD information, skipping")
                                continue
                            try:
                                qlinear = SVDLinear.from_linear_localft(
                                    layer.self_attn.q_proj,
                                    param_ratio=args.rank_ratio,
                                    sigma_fuse='local_ft',
                                    had_rank=args.had_rank,
                                    had_mode='random', # ‘rh'
                                    seed=args.seed,
                                    module_name='q_proj',
                                    is_per_head_svd=args.is_per_head_svd,
                                    is_rank_allocation=args.is_rank_allocate_ft,
                                )
                                
                                rsetattr(layers[idx], 'self_attn.q_proj', qlinear)
                                logging.info(f"Layer {idx} Q fusion SVD completed")
                                klinear = SVDLinear.from_linear_localft(
                                    layer.self_attn.k_proj,
                                    param_ratio=args.rank_ratio,
                                    sigma_fuse='local_ft',
                                    had_rank=args.had_rank,
                                    had_mode='random', # ‘rh'
                                    seed=args.seed,
                                    module_name='k_proj',
                                    is_per_head_svd=args.is_per_head_svd,
                                    is_rank_allocation=args.is_rank_allocate_ft,
                                )
                                
                                rsetattr(layers[idx], 'self_attn.k_proj', klinear)
                                logging.info(f"Layer {idx} K fusion SVD completed")
                                vlinear = SVDLinear.from_linear_localft(
                                    layer.self_attn.v_proj,
                                    param_ratio=args.rank_ratio,
                                    sigma_fuse='local_ft',
                                    had_rank=args.had_rank,
                                    had_mode='random', # ‘rh'
                                    seed=args.seed,
                                    module_name='v_proj',
                                    is_per_head_svd=args.is_per_head_svd,
                                    is_rank_allocation=args.is_rank_allocate_ft,
                                )
                                
                                rsetattr(layers[idx], 'self_attn.v_proj', vlinear)
                                logging.info(f"Layer {idx} V fusion SVD completed")
                            except Exception as e:
                                logging.info(f"Layer {idx} QKV separate fusion SVD failed: {e}")
                                import traceback
                                logging.error(traceback.format_exc())
                    if 'o_proj' in name and args.svd_modules in ['attn', 'all', 'o', 'all_sep_fine', 'all_sep_fine_wo_down', 'all_sep','all_sep_rank_allocate_qkvonly']:
                        if not hasattr(layer.self_attn.o_proj, 'svd_info'):
                            logging.info(f"Layer {idx} o_proj has no pre-calculated SVD information, skipping")
                            continue
                        try:
                            olinear = SVDLinear.from_linear_localft(
                                _unwrap_actquant_linear(layer.self_attn.o_proj),
                                param_ratio=args.rank_ratio,
                                sigma_fuse='local_ft',
                                had_rank=args.had_rank,
                                had_mode='random', # ‘rh'
                                seed=args.seed,
                                module_name='o_proj'
                            )
                            
                            rsetattr(layers[idx], 'self_attn.o_proj', olinear)  
                            logging.info(f"Layer {idx} o_proj fusion SVD completed")
                        except Exception as e:
                            logging.info(f"Layer {idx} o_proj fusion SVD failed: {e}")
                            import traceback
                            logging.error(traceback.format_exc())
                    if 'mlp.up_proj' in name and args.svd_modules in ['all', 'mlp', 'gaup', 'all_sep_fine', 'all_sep_fine_wo_down', 'all_sep','all_sep_rank_allocate_qkvonly']:
                        if not hasattr(layer.mlp.up_proj, 'svd_info'):
                            logging.info(f"Layer {idx} mlp.up_proj has no pre-calculated SVD information, skipping")
                            continue
                        if args.mlp_fuse:
                            try:
                                uplinear, gatedlinear = SVDLinear.from_linearkv_localft(
                                    layer.mlp.up_proj,
                                    layer.mlp.gate_proj,
                                    param_ratio=args.rank_ratio,
                                    sigma_fuse='local_ft',
                                    had_rank=args.had_rank,
                                    had_mode='random', # ‘rh'
                                    seed=args.seed,
                                    module_name='up_proj'
                                )
                                rsetattr(layers[idx], 'mlp.up_proj', uplinear)
                                rsetattr(layers[idx], 'mlp.gate_proj', gatedlinear)
                                logging.info(f"Layer {idx} mlp.up_proj fusion SVD completed")
                            except Exception as e:
                                logging.info(f"Layer {idx} mlp.up_proj fusion SVD failed: {e}")
                                import traceback
                                logging.error(traceback.format_exc())
                        else:
                            try:
                                uplinear = SVDLinear.from_linear_localft(
                                    layer.mlp.up_proj,
                                    param_ratio=args.rank_ratio,
                                    sigma_fuse='local_ft',
                                    had_rank=args.had_rank,
                                    had_mode='random', # ‘rh'
                                    seed=args.seed,
                                    module_name='up_proj'
                                )
                                
                                rsetattr(layers[idx], 'mlp.up_proj', uplinear)
                                logging.info(f"Layer {idx} mlp.up_proj fusion SVD completed")
                                gatedlinear = SVDLinear.from_linear_localft(
                                    layer.mlp.gate_proj,
                                    param_ratio=args.rank_ratio,
                                    sigma_fuse='local_ft',
                                    had_rank=args.had_rank,
                                    had_mode='random', # ‘rh'
                                    seed=args.seed,
                                    module_name='gate_proj'
                                )
                                
                                rsetattr(layers[idx], 'mlp.gate_proj', gatedlinear)
                                logging.info(f"Layer {idx} mlp.gate_proj fusion SVD completed")
                            except Exception as e:
                                logging.info(f"Layer {idx} mlp.up_proj fusion SVD failed: {e}")
                                logging.info(f"Layer {idx} mlp.gate_proj fusion SVD failed: {e}")
                                import traceback
                                logging.error(traceback.format_exc())
                    if 'mlp.down_proj' in name and args.svd_modules in ['all', 'mlp', 'down', 'all_sep_fine', 'all_sep_fine_wo_down', 'all_sep','all_sep_rank_allocate_qkvonly']:
                        if not hasattr(layer.mlp.down_proj, 'svd_info'):
                            logging.info(f"Layer {idx} mlp.down_proj has no pre-calculated SVD information, skipping")
                            continue
                        try:
                            downlinear = SVDLinear.from_linear_localft(
                                _unwrap_actquant_linear(layer.mlp.down_proj),
                                param_ratio=args.rank_ratio,
                                sigma_fuse='local_ft',
                                had_rank=args.had_rank,
                                had_mode='random', # ‘rh'
                                seed=args.seed,
                                module_name='down_proj'
                            )
                            rsetattr(layers[idx], 'mlp.down_proj', downlinear)
                            logging.info(f"Layer {idx} mlp.down_proj fusion SVD completed")
                        except Exception as e:
                            logging.info(f"Layer {idx} mlp.down_proj fusion SVD failed: {e}")
                            import traceback
                            logging.error(traceback.format_exc())


            utils.cleanup_memory(verbos=True)
            return

        else:
            pass
        
        if args.progressive_svd:
            # LSVD release: progressive SVD (utils/progressive_svd_utils.py) is not included.
            raise NotImplementedError("--progressive_svd is not included in the LSVD release.")
        else:
            pass
        
        grad_info_utils.calib_grad_info(
            model=model,
            dataloader=dataloader,
            tokenizer=tokenizer,
            image_processor=image_processor,
            args=args,
            use_cache=args.use_cache,  # default enable cache
            cache_file=None
        )
    
    # Continue performing SVD compression
    model_type = model_utils.get_model_type(model)
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, model_type=model_type)
    svd_debug_S = None

    if args.grad_info:
        rank, world_size = get_rank_and_world_size()
        top_indices, top_scores, layer_indices_dict = grad_info_utils.svd_qkv_with_grad_info(layers, args, use_cache=args.use_cache) # top_indices: (layer_idx, singular_value_idx)
        for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="LM SVD with grad info")):
            full = _find_svd_candidate_layers(layers[idx])
            for name, module in full.items():
                if args.qkv_fuse:  # also controls the mlp-up fuse
                    if 'k_proj' in name and args.svd_modules in ['qkv', 'attn', 'all']: # fuse QKV
                        # Check whether there are pre-calculated SVD information
                        if not hasattr(layer.self_attn.k_proj, 'qkv_svd_info'):
                            logging.info(f"Layer {idx} QKV has no pre-calculated SVD information, skipping")
                            continue
                        
                        # Ensure idx is in layer_indices_dict
                        if idx not in layer_indices_dict:
                            logging.info(f"Layer {idx} QKV is not in layer_indices_dict, skipping")
                            continue
                        
                        # Print more debugging information
                        if rank == 0:
                            logging.info(f"Layer {idx} starting QKV fusion SVD, using {len(layer_indices_dict[idx]['k_proj'])} singular values")
                        
                        
                        try:
                            qlinear, klinear, vlinear = SVDLinear.from_linearqkv_with_grad(
                                layer.self_attn.q_proj,
                                layer.self_attn.k_proj,
                                layer.self_attn.v_proj,
                                param_ratio=args.rank_ratio,
                                alpha=args.act_alpha,
                                act_aware=args.act_aware,
                                rank_align=1.,
                                sigma_fuse=args.svd_mode,
                                had_rank=args.had_rank,
                                had_mode='random', # ‘rh'
                                singular_indices=layer_indices_dict[idx]['k_proj'],
                                seed=args.seed,
                                module_name='k_proj',
                                is_per_head_svd=args.is_per_head_svd,
                                is_q_headnum=args.is_q_headnum,
                            )

                            rsetattr(layers[idx], 'self_attn.q_proj', qlinear)
                            rsetattr(layers[idx], 'self_attn.k_proj', klinear)
                            rsetattr(layers[idx], 'self_attn.v_proj', vlinear)
                            if rank == 0:
                                logging.info(f"Layer {idx} QKV fusion SVD completed")
                        except Exception as e:
                            logging.info(f"Layer {idx} QKV fusion SVD failed: {e}")
                            import traceback
                            logging.error(traceback.format_exc())
                            
                elif args.kv_fuse:
                    # Handle Q separately
                    if 'q_proj' in name and args.svd_modules in ['qkv', 'attn', 'all']:
                        if not hasattr(layer.self_attn.q_proj, 'svd_info'):
                            logging.info(f"Layer {idx} Q has no pre-calculated SVD information, skipping")
                            continue
                        
                        if idx not in layer_indices_dict or 'q_proj' not in layer_indices_dict[idx]:
                            logging.info(f"Layer {idx} Q is not in layer_indices_dict, skipping")
                            continue
                        
                        if rank == 0:
                            logging.info(f"Layer {idx} starting Q SVD, using {len(layer_indices_dict[idx]['q_proj'])} singular values")
                        
                        try:
                            qlinear = SVDLinear.from_linear_with_grad(
                                layer.self_attn.q_proj,
                                param_ratio=args.rank_ratio,
                                alpha=args.act_alpha,
                                act_aware=args.act_aware,
                                loss_aware=args.loss_aware,
                                rank_align=1.,
                                sigma_fuse=args.svd_mode,
                                had_rank=args.had_rank,
                                had_mode='random',
                                singular_indices=layer_indices_dict[idx]['q_proj'],
                                seed=args.seed,
                                module_name='q_proj'
                            )
                            rsetattr(layers[idx], 'self_attn.q_proj', qlinear)
                            if rank == 0:
                                logging.info(f"Layer {idx} Q SVD completed")
                        except Exception as e:
                            logging.info(f"Layer {idx} Q SVD failed: {e}")
                            import traceback
                            logging.error(traceback.format_exc())
                    
                    # Handle KV fusion
                    elif 'k_proj' in name and args.svd_modules in ['qkv', 'attn', 'all']:
                        if not hasattr(layer.self_attn.k_proj, 'svd_info'):
                            logging.info(f"Layer {idx} KV has no pre-calculated SVD information, skipping")
                            continue
                        
                        if idx not in layer_indices_dict or 'k_proj' not in layer_indices_dict[idx]:
                            logging.info(f"Layer {idx} KV is not in layer_indices_dict, skipping")
                            continue
                        
                        if rank == 0:
                            logging.info(f"Layer {idx} starting KV fusion SVD, using {len(layer_indices_dict[idx]['k_proj'])} singular values")
                        
                        try:
                            klinear, vlinear = SVDLinear.from_linearkv_with_grad(
                                layer.self_attn.k_proj,
                                layer.self_attn.v_proj,
                                param_ratio=args.rank_ratio,
                                alpha=args.act_alpha,
                                act_aware=args.act_aware,
                                loss_aware=args.loss_aware,
                                rank_align=1.,
                                sigma_fuse=args.svd_mode,
                                had_rank=args.had_rank,
                                had_mode='random',
                                singular_indices=layer_indices_dict[idx]['k_proj'],
                                seed=args.seed,
                                module_name='k_proj'
                            )
                            rsetattr(layers[idx], 'self_attn.k_proj', klinear)
                            rsetattr(layers[idx], 'self_attn.v_proj', vlinear)
                            if rank == 0:
                                logging.info(f"Layer {idx} KV fusion SVD completed")
                        except Exception as e:
                            logging.info(f"Layer {idx} KV fusion SVD failed: {e}")
                            import traceback
                            logging.error(traceback.format_exc())

                else:
                    # Separate Q, K, V processing
                    if ('q_proj' in name or 'k_proj' in name or 'v_proj' in name) and args.svd_modules in ['qkv', 'attn', 'all']:
                        if 'q_proj' in name:
                            proj_name = 'q_proj'
                            if hasattr(layer.self_attn.k_proj, 'scaling_diag_matrixS'):
                                layer.self_attn.q_proj.scaling_diag_matrixS =  layer.self_attn.k_proj.scaling_diag_matrixS
                                layer.self_attn.v_proj.scaling_diag_matrixS =  layer.self_attn.k_proj.scaling_diag_matrixS
                            elif hasattr(layer.self_attn.k_proj, 'scaling_diag_matrix'):
                                layer.self_attn.q_proj.scaling_diag_matrix =  layer.self_attn.k_proj.scaling_diag_matrix
                                layer.self_attn.v_proj.scaling_diag_matrix =  layer.self_attn.k_proj.scaling_diag_matrix
                            proj = layer.self_attn.q_proj
                        elif 'k_proj' in name:
                            proj_name = 'k_proj'
                            proj = layer.self_attn.k_proj
                        elif 'v_proj' in name:
                            proj_name = 'v_proj'
                            proj = layer.self_attn.v_proj
                        
                        if not hasattr(proj, 'svd_info'):
                            logging.info(f"Layer {idx} {proj_name} has no pre-calculated SVD information, skipping")
                            continue
                        
                        if idx not in layer_indices_dict or proj_name not in layer_indices_dict[idx]:
                            logging.info(f"Layer {idx} {proj_name} is not in layer_indices_dict, skipping")
                            continue
                        
                        if rank == 0:
                            logging.info(f"Layer {idx} starting {proj_name} SVD, using {len(layer_indices_dict[idx][proj_name])} singular values")
                        
                        try:
                            new_linear = SVDLinear.from_linear_with_grad(
                                proj,
                                param_ratio=args.rank_ratio,
                                alpha=args.act_alpha,
                                act_aware=args.act_aware,
                                loss_aware=args.loss_aware,
                                rank_align=1.,
                                sigma_fuse=args.svd_mode,
                                had_rank=args.had_rank,
                                had_mode='random',
                                singular_indices=layer_indices_dict[idx][proj_name],
                                seed=args.seed,
                                module_name=proj_name,
                                is_per_head_svd=args.is_per_head_svd,
                                is_q_headnum=args.is_q_headnum,
                            )
                            rsetattr(layers[idx], f'self_attn.{proj_name}', new_linear)
                            if rank == 0:
                                logging.info(f"Layer {idx} {proj_name} SVD completed")
                        except Exception as e:
                            logging.info(f"Layer {idx} {proj_name} SVD failed: {e}")
                            import traceback
                            logging.error(traceback.format_exc())
                if 'o_proj' in name and args.svd_modules in ['attn', 'all', 'o']:
                    if not hasattr(layer.self_attn.o_proj, 'svd_info'):
                        logging.info(f"Layer {idx} o_proj has no pre-calculated SVD information, skipping")
                        continue
                    
                    if idx not in layer_indices_dict:
                        logging.info(f"Layer {idx} o_proj is not in layer_indices_dict, skipping")
                        continue

                    if rank == 0:
                        logging.info(f"Layer {idx} starting o_proj fusion SVD, using {len(layer_indices_dict[idx]['o_proj'])} singular values")
                    
                    try:
                        olinear = SVDLinear.from_linear_with_grad(
                            layer.self_attn.o_proj,
                            param_ratio=args.rank_ratio,
                            alpha=args.act_alpha,
                            act_aware=args.act_aware,
                            loss_aware=args.loss_aware,
                            rank_align=1.,
                            sigma_fuse=args.svd_mode,
                            had_rank=args.had_rank,
                            had_mode='random', # ‘rh'
                            singular_indices=layer_indices_dict[idx]['o_proj'],
                            seed=args.seed,
                            module_name='o_proj',
                            is_ma_hack=False,
                            topk=1,
                        )
                        rsetattr(layers[idx], 'self_attn.o_proj', olinear)
                        if rank == 0:
                            logging.info(f"Layer {idx} o_proj fusion SVD completed")
                    except Exception as e:
                        logging.info(f"Layer {idx} o_proj fusion SVD failed: {e}")
                        import traceback
                        logging.error(traceback.format_exc())


                if 'mlp.up_proj' in name and args.svd_modules in ['mlp', 'all', 'gaup']:
                    if not hasattr(layer.mlp.up_proj, 'svd_info'):
                        logging.info(f"Layer {idx} mlp.up_proj has no pre-calculated SVD information, skipping")
                        continue
                    
                    if idx not in layer_indices_dict:
                        logging.info(f"Layer {idx} mlp.up_proj is not in layer_indices_dict, skipping")
                    
                    if rank == 0:
                        logging.info(f"Layer {idx} starting mlp.up_proj fusion SVD, using {len(layer_indices_dict[idx]['up_proj'])} singular values")
                    if args.mlp_fuse:
                        try:
                            uplinear, gatedlinear = SVDLinear.from_linearkv_with_grad(
                                layer.mlp.up_proj,
                                layer.mlp.gate_proj,
                                param_ratio=args.rank_ratio,
                                alpha=args.act_alpha,
                                act_aware=args.act_aware,
                                loss_aware=args.loss_aware,
                                rank_align=1.,
                                sigma_fuse=args.svd_mode,
                                had_rank=args.had_rank,
                                had_mode='random', # ‘rh'
                                singular_indices=layer_indices_dict[idx]['up_proj'],
                                seed=args.seed,
                                module_name='up_proj',
                                is_ma_hack=False,
                                topk=1,
                            )
                            rsetattr(layers[idx], 'mlp.up_proj', uplinear)
                            rsetattr(layers[idx], 'mlp.gate_proj', gatedlinear)
                            if rank == 0:
                                logging.info(f"Layer {idx} mlp.up_proj fusion SVD completed")
                        except Exception as e:
                            logging.info(f"Layer {idx} mlp.up_proj fusion SVD failed: {e}")
                            import traceback
                            logging.error(traceback.format_exc())
                    else:
                        layer.mlp.gate_proj.scaling_diag_matrix =  layer.mlp.up_proj.scaling_diag_matrix
                        layer.mlp.gate_proj.scaling_diag_matrixS =  layer.mlp.up_proj.scaling_diag_matrixS
                        try:
                            uplinear = SVDLinear.from_linear_with_grad(
                                layer.mlp.up_proj,
                                param_ratio=args.rank_ratio,
                                alpha=args.act_alpha,
                                act_aware=args.act_aware,
                                loss_aware=args.loss_aware,
                                rank_align=1.,
                                sigma_fuse=args.svd_mode,
                                had_rank=args.had_rank,
                                had_mode='random', # ‘rh'
                                singular_indices=layer_indices_dict[idx]['up_proj'],
                                seed=args.seed,
                                module_name='up_proj',
                                is_ma_hack=False,
                                topk=1,
                            )
                            rsetattr(layers[idx], 'mlp.up_proj', uplinear)
                            if rank == 0:
                                logging.info(f"Layer {idx} mlp.up_proj fusion SVD completed")
                        except Exception as e:
                            logging.info(f"Layer {idx} mlp.up_proj fusion SVD failed: {e}")
                            import traceback
                            logging.error(traceback.format_exc())
                        if rank == 0:
                            
                            logging.info(f"Layer {idx} starting mlp.gate_proj fusion SVD, using {len(layer_indices_dict[idx]['gate_proj'])} singular values")
                        try:
                            gatedlinear = SVDLinear.from_linear_with_grad(
                                layer.mlp.gate_proj,
                                param_ratio=args.rank_ratio,
                                alpha=args.act_alpha,
                                act_aware=args.act_aware,
                                loss_aware=args.loss_aware,
                                rank_align=1.,
                                sigma_fuse=args.svd_mode,
                                had_rank=args.had_rank,
                                had_mode='random', # ‘rh'
                                singular_indices=layer_indices_dict[idx]['gate_proj'],
                                seed=args.seed,
                                module_name='gate_proj',
                                is_ma_hack=False,
                                topk=1,
                            )
                            rsetattr(layers[idx], 'mlp.gate_proj', gatedlinear)
                            if rank == 0:
                                logging.info(f"Layer {idx} mlp.gate_proj fusion SVD completed")
                        except Exception as e:
                            logging.info(f"Layer {idx} mlp.gate_proj fusion SVD failed: {e}")
                            import traceback
                            logging.error(traceback.format_exc())

                if 'mlp.down_proj' in name and args.svd_modules in ['mlp', 'all', 'down']:
                        if not hasattr(layer.mlp.down_proj, 'svd_info'):
                            logging.info(f"Layer {idx} mlp.down_proj has no pre-calculated SVD information, skipping")
                            continue
                        
                        if idx not in layer_indices_dict:
                            logging.info(f"Layer {idx} mlp.down_proj is not in layer_indices_dict, skipping")
                            continue
                        
                        if rank == 0:
                            logging.info(f"Layer {idx} starting mlp.down_proj fusion SVD, using {len(layer_indices_dict[idx]['down_proj'])} singular values")
                        
                        try:
                            downlinear = SVDLinear.from_linear_with_grad(
                                layer.mlp.down_proj,
                                param_ratio=args.rank_ratio,
                                alpha=args.act_alpha,
                                act_aware=args.act_aware,
                                loss_aware=args.loss_aware,
                                rank_align=1.,
                                sigma_fuse=args.svd_mode,
                                had_rank=args.had_rank,
                                had_mode='random', # ‘rh'
                                singular_indices=layer_indices_dict[idx]['down_proj'],
                                seed=args.seed,
                                module_name='down_proj',
                                is_ma_hack=False,
                                topk=1,
                            )
                            rsetattr(layers[idx], 'mlp.down_proj', downlinear)
                            if rank == 0:
                                logging.info(f"Layer {idx} mlp.down_proj fusion SVD completed")
                        except Exception as e:
                            logging.info(f"Layer {idx} mlp.down_proj fusion SVD failed: {e}")
                            import traceback
                            logging.error(traceback.format_exc())
                            

    else:        
        if args.fisher_info:
            rank_allocation = {} # Store rank allocation percentage for each linear layer
            fisher_sum = 0
            fisher_count = 0
            
            # If using qkv_fuse, need to merge Fisher information of q_proj, k_proj and v_proj
            if args.qkv_fuse:
                # First collect Fisher information of q_proj, k_proj and v_proj for each layer
                layer_qkv_fisher = {}
                for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Sum Fisher Info (QKV Fused)")):
                    full = _find_svd_candidate_layers(layers[idx])
                    layer_key = f"layer_{idx}"
                    layer_qkv_fisher[layer_key] = 0
                    
                    for name, module in full.items():
                        if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                            # Add Fisher information of q_proj, k_proj and v_proj together
                            layer_qkv_fisher[layer_key] += module.fisher_info.sum()
                    
                    # Only calculate once per layer (q, k and v merged)
                    if layer_qkv_fisher[layer_key] > 0:
                        fisher_sum += layer_qkv_fisher[layer_key]
                        fisher_count += 1  # Only calculated once per layer, not separately for q, k and v
                
                logging.info(f"Total Fisher Info (QKV Fused): {fisher_sum}, Layer Count: {fisher_count}")
                
                # Calculate rank allocation ratio for each layer
                for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="SVD Rank Allocation (QKV Fused)")):
                    layer_key = f"layer_{idx}"
                    if layer_key in layer_qkv_fisher and fisher_sum > 0:
                        # Calculate the proportion of the layer in total Fisher information
                        layer_ratio = layer_qkv_fisher[layer_key] / fisher_sum * fisher_count
                        # Allocate rank ratio
                        rank_allocation[layer_key] = layer_ratio
                        logging.info(f"Layer {idx} QKV Fused Rank Ratio: {layer_ratio:.4f}")
            elif args.kv_fuse: # If using kv_fuse, need to merge Fisher information of k_proj and v_proj
                # First collect Fisher information of k_proj and v_proj for each layer
                layer_kv_fisher = {}
                for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Sum Fisher Info (KV Fused)")):
                    full = _find_svd_candidate_layers(layers[idx])
                    layer_key = f"layer_{idx}"
                    layer_kv_fisher[layer_key] = 0
                    
                    for name, module in full.items():
                        if 'k_proj' in name or 'v_proj' in name:
                            # Add Fisher information of k_proj and v_proj together
                            layer_kv_fisher[layer_key] += module.fisher_info.sum()
                    
                    # Only calculate once per layer (k and v merged)
                    if layer_kv_fisher[layer_key] > 0:
                        fisher_sum += layer_kv_fisher[layer_key]
                        fisher_count += 1  # Only calculated once per layer, not separately for k and v
                
                logging.info(f"Total Fisher Info (KV Fused): {fisher_sum}, Layer Count: {fisher_count}")
                
                # Calculate rank allocation ratio for each layer
                for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="SVD Rank Allocation (KV Fused)")):
                    layer_key = f"layer_{idx}"
                    if layer_key in layer_kv_fisher and fisher_sum > 0:
                        # Calculate the proportion of the layer in total Fisher information
                        layer_ratio = layer_kv_fisher[layer_key] / fisher_sum * fisher_count
                        # Allocate rank ratio
                        rank_allocation[layer_key] = layer_ratio
                        logging.info(f"Layer {idx} KV Fused Rank Ratio: {layer_ratio:.4f}")
            else:
                # Original non-fusion processing logic
                for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Sum Fisher Info")):
                    full = _find_svd_candidate_layers(layers[idx])
                    for name, module in full.items():
                        # Only apply SVD compression to k_proj and v_proj
                        if 'k_proj' in name or 'v_proj' in name:
                            fisher_sum += module.fisher_info.sum()
                            fisher_count += 1
                
                logging.info(f"Total Fisher Info: {fisher_sum}")
                
                # Calculate rank allocation ratio for each module
                for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="SVD Rank Allocation")):
                    full = _find_svd_candidate_layers(layers[idx])
                    for name, module in full.items():
                        # Only apply SVD compression to k_proj and v_proj
                        if 'k_proj' in name or 'v_proj' in name:
                            # Calculate the proportion of the module in total Fisher information
                            module_ratio = module.fisher_info.sum() / fisher_sum * fisher_count
                            # Allocate rank ratio
                            module_key = f"layer_{idx}_{name}"
                            rank_allocation[module_key] = module_ratio
                            logging.info(f"{module_key} Rank Ratio: {module_ratio:.4f}")
                            
        svd_debug_S = {} if getattr(args, "act_calib_debug", False) else None
        for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="LM SVD")):
            full = _find_svd_candidate_layers(layers[idx])
            for name, module in full.items():
                if args.qkv_fuse:
                    if 'k_proj' in name and args.svd_modules in ['qkv', 'attn', 'all']: # fuse QKV
                        num_kv_heads = layers[idx].self_attn.config.num_key_value_heads if not args.is_q_headnum else layers[idx].self_attn.config.num_attention_heads
                        qlinear, klinear, vlinear = SVDLinear.from_linearqkv(layers[idx].self_attn.q_proj,
                                                        layers[idx].self_attn.k_proj,
                                                        layers[idx].self_attn.v_proj,
                                                        param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}"] if args.fisher_info else args.rank_ratio,
                                                        alpha=args.act_alpha,
                                                        act_aware=args.act_aware,
                                                        loss_aware=args.loss_aware,
                                                        rank_align=1.,
                                                        sigma_fuse=args.svd_mode,
                                                        had_rank=args.had_rank,
                                                        had_mode='random', #'rh'
                                                        seed=args.seed,
                                                        module_name='k_proj',
                                                        true_param_ratio=args.use_true_param_ratio,
                                                        use_group_svd=args.use_group_svd,
                                                        group_ratio=args.group_ratio,
                                                        is_per_head_svd=args.is_per_head_svd,
                                                        is_q_headnum=args.is_q_headnum,
                                                        num_heads=layers[idx].self_attn.config.num_attention_heads,# here only support newer transformers version?
                                                        num_kv_heads=layers[idx].self_attn.config.num_key_value_heads)
                        rsetattr(layers[idx], 'self_attn.q_proj', qlinear)
                        rsetattr(layers[idx], 'self_attn.k_proj', klinear)
                        rsetattr(layers[idx], 'self_attn.v_proj', vlinear)

                elif args.kv_fuse: # Only apply SVD compression to k_proj and v_proj, fuse KV
                    if 'q_proj' in name:
                        layers[idx].self_attn.q_proj.scaling_diag_matrix =  layers[idx].self_attn.k_proj.scaling_diag_matrix
                    if 'k_proj' in name:
                        klinear, vlinear = SVDLinear.from_linearkv(layers[idx].self_attn.k_proj,
                                                        layers[idx].self_attn.v_proj,
                                                        param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}"] if args.fisher_info else args.rank_ratio,
                                                        alpha=args.act_alpha,
                                                        act_aware=args.act_aware,
                                                        loss_aware=args.loss_aware,
                                                        rank_align=1.,
                                                        sigma_fuse=args.svd_mode,
                                                        had_rank=args.had_rank,
                                                        had_mode='random',
                                                        seed=args.seed,
                                                        module_name='k_proj',
                                                        true_param_ratio=args.use_true_param_ratio)
                        rsetattr(layers[idx], 'self_attn.k_proj', klinear)
                        rsetattr(layers[idx], 'self_attn.v_proj', vlinear)
                    elif 'q_proj' in name:
                        rsetattr(layers[idx], name, SVDLinear.from_linear(layers[idx].self_attn.q_proj,
                                                        param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}_{name}"] if args.fisher_info else args.rank_ratio,
                                                        alpha=args.act_alpha,
                                                        act_aware=args.act_aware,
                                                        loss_aware=args.loss_aware,
                                                        sigma_fuse=args.svd_mode,
                                                        rank_align=1.,
                                                        had_rank=args.had_rank,
                                                        had_mode='random',
                                                        seed=args.seed,
                                                        module_name='q_proj',
                                                        svd_debug_store=svd_debug_S,
                                                        svd_debug_key=(f"model.layers.{idx}.{name}" if svd_debug_S is not None else None)))
                else:
                    if ('k_proj' in name or 'v_proj' in name or 'q_proj' in name) and args.svd_modules in ['qkv', 'attn', 'all']: # Only apply SVD compression to k_proj and v_proj
                        if 'q_proj' in name:
                            try:
                                layers[idx].self_attn.q_proj.scaling_diag_matrix =  layers[idx].self_attn.k_proj.scaling_diag_matrix
                                layers[idx].self_attn.v_proj.scaling_diag_matrix =  layers[idx].self_attn.k_proj.scaling_diag_matrix
                            except:
                                layers[idx].self_attn.q_proj.scaling_diag_matrixS =  layers[idx].self_attn.k_proj.scaling_diag_matrixS
                                layers[idx].self_attn.v_proj.scaling_diag_matrixS =  layers[idx].self_attn.k_proj.scaling_diag_matrixS
                                logging.info(f"Layer {idx} q_proj and v_proj scaling_diag_matrixS copied")
                        if 'q_proj' in name:
                            module_name = 'q_proj'
                        elif 'k_proj' in name:
                            module_name = 'k_proj'
                        elif 'v_proj' in name:
                            module_name = 'v_proj'

                        rsetattr(layers[idx], name, SVDLinear.from_linear(_unwrap_actquant_linear(module),
                                                        param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}_{name}"] if args.fisher_info else args.rank_ratio,
                                                        alpha=args.act_alpha,
                                                        act_aware=args.act_aware,
                                                        loss_aware=args.loss_aware,
                                                        sigma_fuse=args.svd_mode,# if 'k_proj' in name else "U",
                                                        rank_align=1.,
                                                        had_rank=args.had_rank,
                                                        had_mode='random',
                                                        seed=args.seed,
                                                        module_name=module_name,
                                                        is_per_head_svd=args.is_per_head_svd,
                                                        is_q_headnum=args.is_q_headnum,
                                                        num_heads=layers[idx].self_attn.config.num_attention_heads,
                                                        num_kv_heads=layers[idx].self_attn.config.num_key_value_heads,
                                                        head_count_override=(
                                                            layers[idx].self_attn.config.num_attention_heads
                                                            if args.is_per_head_svd and 'q_proj' in name
                                                            else layers[idx].self_attn.config.num_key_value_heads
                                                            if args.is_per_head_svd and ('k_proj' in name or 'v_proj' in name)
                                                            else None
                                                        ),
                                                        svd_debug_store=svd_debug_S,
                                                        svd_debug_key=(f"model.layers.{idx}.{name}" if svd_debug_S is not None else None)))
                if 'o_proj' in name and args.svd_modules in ['attn', 'all', 'o']:
                    olinear = SVDLinear.from_linear(_unwrap_actquant_linear(layers[idx].self_attn.o_proj),
                                                    param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}"] if args.fisher_info else args.rank_ratio,
                                                    alpha=args.act_alpha,
                                                    act_aware=args.act_aware,
                                                    loss_aware=args.loss_aware,
                                                    rank_align=1.,
                                                    sigma_fuse=args.svd_mode,
                                                    had_rank=args.had_rank,
                                                    had_mode='random', #'rh'
                                                    seed=args.seed,
                                                    module_name='o_proj',
                                                    svd_debug_store=svd_debug_S,
                                                    svd_debug_key=(f"model.layers.{idx}.{name}" if svd_debug_S is not None else None))
                    rsetattr(layers[idx], 'self_attn.o_proj', olinear)
                if 'mlp.up_proj' in name and args.svd_modules in ['mlp', 'all', 'gaup']:
                    if args.mlp_fuse:
                        uplinear, gatedlinear = SVDLinear.from_linearkv(layers[idx].mlp.up_proj,
                                                        layers[idx].mlp.gate_proj,
                                                        param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}"] if args.fisher_info else args.rank_ratio,
                                                        alpha=args.act_alpha,
                                                        act_aware=args.act_aware,
                                                        loss_aware=args.loss_aware,
                                                        rank_align=1.,
                                                        sigma_fuse=args.svd_mode,
                                                        had_rank=args.had_rank,
                                                        had_mode='random', #'rh'
                                                        seed=args.seed,
                                                        module_name='up_proj',
                                                        true_param_ratio=args.use_true_param_ratio)
                    else:
                        uplinear = SVDLinear.from_linear(layers[idx].mlp.up_proj,
                                                        param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}"] if args.fisher_info else args.rank_ratio,
                                                        alpha=args.act_alpha,
                                                        act_aware=args.act_aware,
                                                        loss_aware=args.loss_aware,
                                                        rank_align=1.,
                                                        sigma_fuse=args.svd_mode,
                                                        had_rank=args.had_rank,
                                                        had_mode='random', #'rh'
                                                        seed=args.seed,
                                                        module_name='up_proj',
                                                        svd_debug_store=svd_debug_S,
                                                        svd_debug_key=(f"model.layers.{idx}.{name}" if svd_debug_S is not None else None))
                        if args.act_aware:
                            try:                                                        
                                layers[idx].mlp.gate_proj.scaling_diag_matrix =  layers[idx].mlp.up_proj.scaling_diag_matrix
                            except:
                                layers[idx].mlp.gate_proj.scaling_diag_matrixS =  layers[idx].mlp.up_proj.scaling_diag_matrixS # add support for cholesky
                        gatedlinear= SVDLinear.from_linear(layers[idx].mlp.gate_proj,
                                                        param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}"] if args.fisher_info else args.rank_ratio,
                                                        alpha=args.act_alpha,
                                                        act_aware=args.act_aware,
                                                        loss_aware=args.loss_aware,
                                                        rank_align=1.,
                                                        sigma_fuse=args.svd_mode,
                                                        had_rank=args.had_rank,
                                                        had_mode='random', #'rh'
                                                        seed=args.seed,
                                                        module_name='gate_proj',
                                                        svd_debug_store=svd_debug_S,
                                                        svd_debug_key=(f"model.layers.{idx}.{name}" if svd_debug_S is not None else None))
                    rsetattr(layers[idx], 'mlp.up_proj', uplinear)
                    rsetattr(layers[idx], 'mlp.gate_proj', gatedlinear)
                if 'mlp.down_proj' in name and args.svd_modules in ['mlp', 'all', 'down']:
                    if args.is_ma_hack and idx  in [1, 31]:
                        downlinear = SVDLinear.from_linear(_unwrap_actquant_linear(layers[idx].mlp.down_proj),
                            param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}"] if args.fisher_info else args.rank_ratio,
                            alpha=args.act_alpha,
                            act_aware=args.act_aware,
                            loss_aware=args.loss_aware,
                            rank_align=1.,
                            sigma_fuse=args.svd_mode,
                            had_rank=args.had_rank,
                            had_mode='random', #'rh'
                            seed=args.seed,
                            module_name='down_proj',
                            is_ma_hack=True,
                            topk=args.topk,
                            svd_debug_store=svd_debug_S,
                            svd_debug_key=(f"model.layers.{idx}.{name}" if svd_debug_S is not None else None),
                        )
                        logging.info(f"Layer {idx} mlp.down_proj fake MA hack fusion SVD completed")
                    else:
                        downlinear = SVDLinear.from_linear(_unwrap_actquant_linear(layers[idx].mlp.down_proj),
                                                        param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}"] if args.fisher_info else args.rank_ratio,
                                                        alpha=args.act_alpha,
                                                        act_aware=args.act_aware,
                                                        loss_aware=args.loss_aware,
                                                        rank_align=1.,
                                                        sigma_fuse=args.svd_mode,
                                                        had_rank=args.had_rank,
                                                        had_mode='random', #'rh'
                                                        seed=args.seed,
                                                        module_name='down_proj',
                                                        svd_debug_store=svd_debug_S,
                                                        svd_debug_key=(f"model.layers.{idx}.{name}" if svd_debug_S is not None else None))
                    rsetattr(layers[idx], 'mlp.down_proj', downlinear)

        if svd_debug_S:
            debug_dir = getattr(args, "act_calib_debug_dir", None) or getattr(args, "save_path", ".")
            os.makedirs(debug_dir, exist_ok=True)
            _sn = getattr(args, "save_name", "debug")
            svd_out = os.path.join(debug_dir, f"act_calib_debug_svd_S_{_sn}.pt")
            torch.save(svd_debug_S, svd_out)
            logging.info(
                "act_calib_debug: saved full SVD singular value tensors for %d modules to %s",
                len(svd_debug_S),
                svd_out,
            )
    if args.had_rank:
        logging.info("Starting to apply had_rank")
        rotation_utils.had_transform_rank(model)
    for module_name, rank_sum in total_rank_sum.items():
        if rank_sum > 0:
            if args.is_per_head_svd and 'k_proj' in module_name:
                num_kv_heads = layers[0].self_attn.config.num_key_value_heads if not args.is_q_headnum else layers[0].self_attn.config.num_attention_heads
                logging.info(f"Language model SVD compression completed with {module_name} total rank sum: {rank_sum} across {total_linear_count[module_name]} linear layers, total KV head num: {num_kv_heads}, per head rank: {rank_sum / num_kv_heads}")
            else:
                logging.info(f"Language model SVD compression completed with {module_name} total rank sum: {rank_sum} across {total_linear_count[module_name]} linear layers")

def cap_and_redistribute_rank_allocation(rank_allocation: dict, cap_ratio: float) -> dict:
    """
    Cap each layer's rank ratio at cap_ratio, and redistribute excess proportionally
    to layers that were not capped.

    Args:
        rank_allocation (dict): layer_name -> raw rank ratio
        cap_ratio (float): maximum allowed ratio per layer (e.g., 1.0)

    Returns:
        dict: layer_name -> capped and redistributed rank ratio
    """
    raw_ratios = rank_allocation.copy()
    clipped_ratios = {}
    excess = 0.0

    # First pass: clip ratios and accumulate excess
    for key, ratio in raw_ratios.items():
        if ratio > cap_ratio:
            clipped_ratios[key] = cap_ratio
            excess += ratio - cap_ratio
        else:
            clipped_ratios[key] = ratio

    # Get layers eligible to receive redistribution
    redistribute_keys = [k for k in raw_ratios if raw_ratios[k] <= cap_ratio]
    redistribute_total = sum(raw_ratios[k] for k in redistribute_keys)

    # Redistribute the excess
    for k in redistribute_keys:
        if redistribute_total > 0:
            share = raw_ratios[k] / redistribute_total
            clipped_ratios[k] += share * excess

    return clipped_ratios

def cap_rank_allocation(rank_allocation: dict, cap_ratio: float) -> dict:
    """
    Cap each layer's rank ratio at cap_ratio, and redistribute excess proportionally
    to layers that were not capped.

    Args:
        rank_allocation (dict): layer_name -> raw rank ratio
        cap_ratio (float): maximum allowed ratio per layer (e.g., 1.0)

    Returns:
        dict: layer_name -> capped and redistributed rank ratio
    """
    raw_ratios = rank_allocation.copy()
    clipped_ratios = {}
    excess = 0.0

    # First pass: clip ratios and accumulate excess
    for key, ratio in raw_ratios.items():
        if ratio > cap_ratio:
            clipped_ratios[key] = cap_ratio

    return clipped_ratios

def svd_emb(model, args):
    embedding = model_utils.get_embeddings(model, model_utils.get_model_type(model))[0] # nn.Embedding
    w = embedding.weight.data
    dtype = embedding.weight.dtype
    U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False)
    if args.use_true_param_ratio or args.use_param_ratio:
        rank = int(args.rank_ratio * embedding.weight.numel() / (embedding.num_embeddings + embedding.embedding_dim))
    else:
        rank = int (args.rank_ratio/2 * min(embedding.num_embeddings, embedding.embedding_dim))
    U = U[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]
    up = U.mul(S.sqrt()).contiguous() # 32000, 2560
    down = Vt.mul(S.sqrt().view(-1, 1)).contiguous() # 2560, 5120
    up_linear = nn.Linear(rank, embedding.embedding_dim, bias=False)
    down_linear = nn.Embedding(embedding.num_embeddings, rank, embedding.padding_idx)
    logging.info(f'SVD embedding rank: {rank}, h_dim: {min(embedding.num_embeddings, embedding.embedding_dim)}, rank_ratio: {rank/min(embedding.num_embeddings, embedding.embedding_dim)}')
    up_linear.weight.data = down.t()
    down_linear.weight.data = up # here embedding is not like linear layers
    embedding = torch.nn.Sequential(
        down_linear.to(dtype),
        up_linear.to(dtype)
    )
    model.model.embed_tokens = embedding.to(dtype)
    
    