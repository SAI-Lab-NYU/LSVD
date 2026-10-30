import os
import logging
import hashlib

import torch


def safe_inverse_cpu(mat_cpu_double):
    try:
        return torch.linalg.inv(mat_cpu_double).float()
    except Exception as e:
        logging.warning("act_calib_debug: torch.linalg.inv failed (%s), using pinv", e)
        return torch.linalg.pinv(mat_cpu_double).float()


def covariance_debug_summary(
    raw_cpu_double,
    chol_cpu,
    normalization_count,
    sample_count,
    token_count,
    hook_count,
    diag_shift_cpu=None,
    chol_is_upper=False,
):
    raw = raw_cpu_double.double()
    raw_sym = (raw + raw.transpose(-1, -2)) / 2
    norm_div = float(max(float(normalization_count), 1.0))
    norm = raw_sym / norm_div

    sym_err_raw = torch.linalg.norm(raw - raw.transpose(-1, -2), dim=(-2, -1)).float()
    sym_err_norm = torch.linalg.norm(norm - norm.transpose(-1, -2), dim=(-2, -1)).float()

    eig_raw = torch.linalg.eigvalsh(raw_sym).float()
    eig_norm = torch.linalg.eigvalsh(norm).float()
    min_eig_raw = eig_raw[..., 0]
    max_eig_raw = eig_raw[..., -1]
    min_eig_norm = eig_norm[..., 0]
    max_eig_norm = eig_norm[..., -1]
    eps = 1e-30
    cond_est_raw = max_eig_raw / min_eig_raw.abs().clamp_min(eps)
    cond_est_norm = max_eig_norm / min_eig_norm.abs().clamp_min(eps)

    chol_diag = torch.diagonal(chol_cpu.float(), dim1=-2, dim2=-1)
    chol_diag_min = chol_diag.amin(dim=-1)
    chol_diag_max = chol_diag.amax(dim=-1)

    inv_raw = safe_inverse_cpu(raw_sym).double()
    inv_norm_mat = safe_inverse_cpu(norm).double()
    inv_norm = torch.linalg.norm(inv_raw, dim=(-2, -1)).float()

    if chol_is_upper:
        chol_recon_raw = chol_cpu.double().transpose(-1, -2) @ chol_cpu.double()
    else:
        chol_recon_raw = chol_cpu.double() @ chol_cpu.double().transpose(-1, -2)
    chol_recon_norm = chol_recon_raw / norm_div

    eye = torch.eye(raw_sym.shape[-1], dtype=raw_sym.dtype, device=raw_sym.device)
    if raw_sym.ndim > 2:
        eye = eye.expand(raw_sym.shape[:-2] + eye.shape)

    chol_abs_err_raw = torch.linalg.norm(raw_sym - chol_recon_raw, dim=(-2, -1)).float()
    chol_rel_err_raw = chol_abs_err_raw / torch.linalg.norm(raw_sym, dim=(-2, -1)).float().clamp_min(1e-30)
    chol_abs_err_norm = torch.linalg.norm(norm - chol_recon_norm, dim=(-2, -1)).float()
    chol_rel_err_norm = chol_abs_err_norm / torch.linalg.norm(norm, dim=(-2, -1)).float().clamp_min(1e-30)
    inv_abs_err_raw = torch.linalg.norm(raw_sym @ inv_raw - eye, dim=(-2, -1)).float()
    inv_rel_err_raw = inv_abs_err_raw / torch.linalg.norm(eye, dim=(-2, -1)).float().clamp_min(1e-30)
    inv_abs_err_norm = torch.linalg.norm(norm @ inv_norm_mat - eye, dim=(-2, -1)).float()
    inv_rel_err_norm = inv_abs_err_norm / torch.linalg.norm(eye, dim=(-2, -1)).float().clamp_min(1e-30)

    diag_shift = (
        diag_shift_cpu.float()
        if isinstance(diag_shift_cpu, torch.Tensor)
        else torch.tensor(float(diag_shift_cpu or 0.0), dtype=torch.float32)
    )

    return {
        "sample_count": int(sample_count),
        "token_count": int(token_count),
        "hook_count": int(hook_count),
        "normalization_count": float(norm_div),
        "sym_err_raw": sym_err_raw.cpu(),
        "min_eig_raw": min_eig_raw.cpu(),
        "max_eig_raw": max_eig_raw.cpu(),
        "cond_est_raw": cond_est_raw.cpu(),
        "sym_err_norm": sym_err_norm.cpu(),
        "min_eig_norm": min_eig_norm.cpu(),
        "max_eig_norm": max_eig_norm.cpu(),
        "cond_est_norm": cond_est_norm.cpu(),
        "diag_shift": diag_shift.cpu(),
        "chol_diag_min": chol_diag_min.cpu(),
        "chol_diag_max": chol_diag_max.cpu(),
        "inv_norm": inv_norm.cpu(),
        "chol_abs_err_raw": chol_abs_err_raw.cpu(),
        "chol_rel_err_raw": chol_rel_err_raw.cpu(),
        "chol_abs_err_norm": chol_abs_err_norm.cpu(),
        "chol_rel_err_norm": chol_rel_err_norm.cpu(),
        "inv_abs_err_raw": inv_abs_err_raw.cpu(),
        "inv_rel_err_raw": inv_rel_err_raw.cpu(),
        "inv_abs_err_norm": inv_abs_err_norm.cpu(),
        "inv_rel_err_norm": inv_rel_err_norm.cpu(),
    }


def act_calib_debug_pack(raw_cpu_double, chol_cpu):
    return {
        "input_covariance": raw_cpu_double.float(),
        "cholesky_L": chol_cpu.float(),
        "covariance_inverse": safe_inverse_cpu(raw_cpu_double),
    }


def act_calib_debug_pack_with_summary(
    raw_cpu_double,
    chol_cpu,
    normalization_count,
    sample_count,
    token_count,
    hook_count,
    diag_shift_cpu=None,
):
    payload = act_calib_debug_pack(raw_cpu_double, chol_cpu)
    norm_div = float(max(float(normalization_count), 1.0))
    raw_sym = ((raw_cpu_double.double() + raw_cpu_double.double().transpose(-1, -2)) / 2).cpu()
    payload["input_covariance_normalized"] = (raw_sym / norm_div).float()
    payload["covariance_inverse_normalized"] = safe_inverse_cpu(raw_sym / norm_div)
    payload["summary"] = covariance_debug_summary(
        raw_cpu_double=raw_cpu_double.cpu(),
        chol_cpu=chol_cpu.cpu(),
        normalization_count=norm_div,
        sample_count=sample_count,
        token_count=token_count,
        hook_count=hook_count,
        diag_shift_cpu=diag_shift_cpu.cpu() if isinstance(diag_shift_cpu, torch.Tensor) else diag_shift_cpu,
    )
    return payload


def grad_calib_debug_pack(
    raw_cpu_double,
    chol_cpu,
    normalization_count,
    sample_count,
    token_count,
    hook_count,
    diag_shift_cpu=None,
):
    raw_sym = ((raw_cpu_double.double() + raw_cpu_double.double().transpose(-1, -2)) / 2).cpu()
    norm_div = float(max(float(normalization_count), 1.0))
    return {
        "output_grad_covariance": raw_sym.float(),
        "output_grad_covariance_normalized": (raw_sym / norm_div).float(),
        "cholesky_U": chol_cpu.float(),
        "covariance_inverse": safe_inverse_cpu(raw_sym),
        "covariance_inverse_normalized": safe_inverse_cpu(raw_sym / norm_div),
        "summary": covariance_debug_summary(
            raw_cpu_double=raw_cpu_double.cpu(),
            chol_cpu=chol_cpu.cpu(),
            normalization_count=norm_div,
            sample_count=sample_count,
            token_count=token_count,
            hook_count=hook_count,
            diag_shift_cpu=diag_shift_cpu.cpu() if isinstance(diag_shift_cpu, torch.Tensor) else diag_shift_cpu,
            chol_is_upper=True,
        ),
    }


def save_act_calib_debug_payload(args, filename, payload, log_prefix):
    if not payload:
        return
    debug_dir = getattr(args, "act_calib_debug_dir", None) or getattr(args, "save_path", ".")
    os.makedirs(debug_dir, exist_ok=True)
    save_name = getattr(args, "save_name", "debug")
    dbg_path = os.path.join(debug_dir, f"{filename}_{save_name}.pt")
    if os.path.exists(dbg_path):
        try:
            existing = torch.load(dbg_path, map_location="cpu", weights_only=False)
            if isinstance(existing, dict):
                merged = dict(existing)
                for key, value in payload.items():
                    if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                        merged[key] = {**merged[key], **value}
                    else:
                        merged[key] = value
                payload = merged
        except Exception as e:
            logging.warning(
                "act_calib_debug: failed to merge existing payload from %s (%s); overwriting",
                dbg_path,
                e,
            )
    torch.save(payload, dbg_path)
    try:
        n_bytes = os.path.getsize(dbg_path)
    except OSError:
        n_bytes = 0
    logging.info(
        "%s: saved %d entries to %s (~%.2f MiB)",
        log_prefix,
        len(payload),
        dbg_path,
        n_bytes / (1024 * 1024),
    )


def summarize_singular_values(S):
    s = S.detach().float().cpu()
    if s.ndim == 1:
        top1 = s[0]
        energy = torch.sum(s ** 2)
        topk = min(8, s.shape[0])
        ratio = torch.sum(s[:topk] ** 2) / energy.clamp_min(1e-30)
        return top1, ratio
    top1 = s[..., 0]
    energy = torch.sum(s ** 2, dim=-1)
    topk = min(8, s.shape[-1])
    ratio = torch.sum(s[..., :topk] ** 2, dim=-1) / energy.clamp_min(1e-30)
    return top1, ratio


def record_prepare_svd_debug(
    debug_store,
    key,
    weight_before,
    weight_after,
    S,
    output_grad_scaling_matrix_inv=None,
    input_inverse=None,
):
    if debug_store is None:
        return

    def _stable_norm(x):
        x = x.detach().float()
        if x.ndim >= 2:
            return torch.linalg.norm(x, dim=(-2, -1)).cpu()
        return torch.linalg.norm(x).cpu()

    sv_top1, sv_topk_ratio = summarize_singular_values(S)
    payload = {
        "weight_norm": torch.linalg.norm(weight_before.detach().float(), dim=(-2, -1)).cpu(),
        "whitened_weight_norm": torch.linalg.norm(weight_after.detach().float(), dim=(-2, -1)).cpu(),
        "sv_top1": sv_top1.cpu(),
        "sv_topk_energy_ratio": sv_topk_ratio.cpu(),
    }
    if S.detach().float().ndim >= 2:
        payload["sv_top1_head"] = sv_top1.cpu()
        payload["sv_topk_energy_ratio_head"] = sv_topk_ratio.cpu()
        payload["weight_norm_head"] = torch.linalg.norm(weight_before.detach().float(), dim=(-2, -1)).cpu()
        payload["whitened_weight_norm_head"] = torch.linalg.norm(weight_after.detach().float(), dim=(-2, -1)).cpu()
    if output_grad_scaling_matrix_inv is not None:
        payload["inv_norm"] = _stable_norm(output_grad_scaling_matrix_inv)
    elif input_inverse is not None:
        payload["inv_norm"] = _stable_norm(input_inverse)
    debug_store[key] = payload


def summarize_vector_metric(x, topk=8):
    x = x.detach().float().cpu()
    abs_x = x.abs()
    if abs_x.numel() == 0:
        zero = torch.tensor(0.0, dtype=torch.float32)
        return {
            "sum": zero,
            "mean": zero,
            "top1": zero,
            "topk_ratio": zero,
        }
    flat = abs_x.reshape(-1)
    topk = min(int(topk), flat.numel())
    total = flat.sum().clamp_min(1e-30)
    return {
        "sum": flat.sum(),
        "mean": flat.mean(),
        "top1": flat.max(),
        "topk_ratio": flat.topk(topk).values.sum() / total,
    }


def importance_debug_pack(singular_values, s_grad, importance_score):
    sv = singular_values.detach().float().cpu()
    sg = s_grad.detach().float().cpu()
    imp = importance_score.detach().float().cpu()
    payload = {
        "singular_values": sv,
        "s_grad": sg,
        "importance_score": imp,
        "importance_sum": summarize_vector_metric(imp)["sum"],
        "importance_mean": summarize_vector_metric(imp)["mean"],
        "importance_top1": summarize_vector_metric(imp)["top1"],
        "importance_topk_ratio": summarize_vector_metric(imp)["topk_ratio"],
        "singular_top1": summarize_vector_metric(sv)["top1"],
        "s_grad_top1": summarize_vector_metric(sg)["top1"],
    }
    if imp.ndim >= 2:
        payload["importance_head_sum"] = imp.abs().sum(dim=-1)
        payload["importance_head_top1"] = imp.abs().amax(dim=-1)
    return payload


def attach_importance_selection(payload, selected_indices, allocated_rank_ratio=None):
    if payload is None:
        return
    imp = payload.get("importance_score")
    if not isinstance(imp, torch.Tensor):
        return

    selected_count = 0
    selected_scores = None
    selected_head_count = None

    if isinstance(selected_indices, dict):
        gathered = []
        head_counts = []
        for head_idx in sorted(selected_indices):
            idxs = list(selected_indices[head_idx])
            head_counts.append(len(idxs))
            selected_count += len(idxs)
            if imp.ndim >= 2 and len(idxs) > 0:
                gathered.append(imp[int(head_idx), idxs])
        if gathered:
            selected_scores = torch.cat([g.reshape(-1) for g in gathered], dim=0)
        selected_head_count = torch.tensor(head_counts, dtype=torch.int32)
    else:
        idxs = list(selected_indices or [])
        selected_count = len(idxs)
        if len(idxs) > 0:
            selected_scores = imp[idxs]

    payload["selected_indices"] = selected_indices
    payload["selected_indices_by_head"] = selected_indices if isinstance(selected_indices, dict) else None
    payload["selected_count"] = int(selected_count)
    payload["allocated_rank_ratio"] = (
        None if allocated_rank_ratio is None else float(allocated_rank_ratio)
    )
    available_count = int(imp.shape[-1]) if imp.ndim >= 1 else int(imp.numel())
    payload["available_count"] = available_count
    payload["selected_ratio"] = float(selected_count / max(available_count, 1))
    if selected_scores is not None and isinstance(selected_scores, torch.Tensor) and selected_scores.numel() > 0:
        stats = summarize_vector_metric(selected_scores)
        payload["selected_scores"] = selected_scores.detach().float().cpu()
        payload["selected_score_sum"] = stats["sum"]
        payload["selected_score_mean"] = stats["mean"]
        payload["selected_score_top1"] = stats["top1"]
        payload["selected_score_topk_ratio"] = stats["topk_ratio"]
    if selected_head_count is not None:
        payload["selected_head_count"] = selected_head_count
        payload["selected_head_ratio"] = selected_head_count.float() / max(available_count, 1)


def attach_selection_debug(payload, selected_indices, available_count=None, allocated_rank_ratio=None, selected_scores=None):
    if payload is None:
        return
    selected_count = 0
    selected_head_count = None
    if isinstance(selected_indices, dict):
        head_counts = []
        for head_idx in sorted(selected_indices):
            idxs = list(selected_indices[head_idx])
            head_counts.append(len(idxs))
            selected_count += len(idxs)
        selected_head_count = torch.tensor(head_counts, dtype=torch.int32)
    else:
        selected_count = len(list(selected_indices or []))

    payload["selected_indices"] = selected_indices
    payload["selected_indices_by_head"] = selected_indices if isinstance(selected_indices, dict) else None
    payload["selected_count"] = int(selected_count)
    if available_count is not None:
        payload["available_count"] = int(available_count)
        payload["selected_ratio"] = float(selected_count / max(int(available_count), 1))
    if allocated_rank_ratio is not None:
        payload["allocated_rank_ratio"] = float(allocated_rank_ratio)
    if selected_head_count is not None:
        payload["selected_head_count"] = selected_head_count
        if available_count is not None:
            payload["selected_head_ratio"] = selected_head_count.float() / max(int(available_count), 1)
    if isinstance(selected_scores, torch.Tensor) and selected_scores.numel() > 0:
        stats = summarize_vector_metric(selected_scores)
        payload["selected_scores"] = selected_scores.detach().float().cpu()
        payload["selected_score_sum"] = stats["sum"]
        payload["selected_score_mean"] = stats["mean"]
        payload["selected_score_top1"] = stats["top1"]
        payload["selected_score_topk_ratio"] = stats["topk_ratio"]


def attach_floor_debug(
    payload,
    floor_enabled=False,
    floor_target=None,
    floor_before_head_count=None,
    floor_after_head_count=None,
    floor_added_count=None,
    floor_removed_count=None,
    floor_unsatisfied=None,
    floor_cross_layer_enabled=None,
):
    if payload is None:
        return
    payload["floor_enabled"] = bool(floor_enabled)
    if floor_target is not None:
        payload["floor_target"] = int(floor_target)
    if floor_before_head_count is not None:
        payload["floor_before_head_count"] = torch.as_tensor(floor_before_head_count, dtype=torch.int32).cpu()
    if floor_after_head_count is not None:
        payload["floor_after_head_count"] = torch.as_tensor(floor_after_head_count, dtype=torch.int32).cpu()
    if floor_added_count is not None:
        payload["floor_added_count"] = int(floor_added_count)
    if floor_removed_count is not None:
        payload["floor_removed_count"] = int(floor_removed_count)
    if floor_unsatisfied is not None:
        payload["floor_unsatisfied"] = bool(floor_unsatisfied)
    if floor_cross_layer_enabled is not None:
        payload["floor_cross_layer_enabled"] = bool(floor_cross_layer_enabled)


def attach_budget_stage_debug(payload, stage_name, layer_budget=None, head_budget=None, total_budget=None):
    if payload is None or not stage_name:
        return
    stage_payload = payload.setdefault("budget_debug", {}).setdefault(stage_name, {})
    if layer_budget is not None:
        stage_payload["layer_budget"] = int(layer_budget)
    if head_budget is not None:
        stage_payload["head_budget"] = torch.as_tensor(head_budget, dtype=torch.int32).cpu()
    if total_budget is not None:
        stage_payload["total_budget"] = int(total_budget)


def stable_projection_seed(base_seed, key):
    token = f"{int(base_seed)}::{key}".encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2 ** 31)


def make_debug_projection(rows, cols, seed, device=None, dtype=torch.float32):
    rows = int(rows)
    cols = int(cols)
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid projection shape rows={rows}, cols={cols}")
    cols = min(rows, cols)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    raw = torch.randn(rows, cols, generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    q = q.to(dtype=dtype)
    if device is not None:
        q = q.to(device=device)
    return q


def empirical_fisher_debug_pack(config):
    if not isinstance(config, dict):
        return {}
    heads = {}
    for head_idx, head_state in sorted((config.get("heads") or {}).items()):
        heads[int(head_idx)] = {
            "sample_count": int(head_state.get("sample_count", 0)),
            "sketch_outer": torch.as_tensor(head_state.get("sketch_outer"), dtype=torch.float32).cpu(),
            "sketch_diag": torch.as_tensor(head_state.get("sketch_diag"), dtype=torch.float32).cpu(),
            "sketch_trace_sum": float(head_state.get("sketch_trace_sum", 0.0)),
            "output_projection": torch.as_tensor(head_state.get("output_projection"), dtype=torch.float32).cpu(),
            "projection_seed": int(head_state.get("projection_seed", 0)),
        }
    return {
        "enabled": True,
        "selection_strategy": str(config.get("selection_strategy", "")),
        "selected_head_indices": torch.as_tensor(config.get("selected_head_indices", []), dtype=torch.int32).cpu(),
        "selected_layer_idx": int(config.get("selected_layer_idx", -1)),
        "head_dim": int(config.get("head_dim", 0)),
        "in_features": int(config.get("in_features", 0)),
        "input_proj_dim": int(config.get("input_proj_dim", 0)),
        "output_proj_dim": int(config.get("output_proj_dim", 0)),
        "input_projection": torch.as_tensor(config.get("input_projection"), dtype=torch.float32).cpu(),
        "input_projection_seed": int(config.get("input_projection_seed", 0)),
        "sample_count": int(config.get("sample_count", 0)),
        "sketch_dim": int(config.get("sketch_dim", 0)),
        "heads": heads,
    }
