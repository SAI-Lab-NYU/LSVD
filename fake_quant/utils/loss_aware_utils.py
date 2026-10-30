import os
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import logging
from . import (
    utils,
    gptq_utils,
    data_utils,
    model_utils,
    grad_info_utils,
    debug_utils,
    molmo_utils,
    reasoning_calib_utils,
    calib_cache,
)


def _is_lm_layer_name(name):
    return (
        "model.layers" in name
        or "language_model.layers" in name
        or "text_model.layers" in name
        or "transformer.blocks" in name
    )


def _parse_rank_floor_layers(layer_spec, num_layers):
    if layer_spec is None:
        return set()
    selected = set()
    for part in str(layer_spec).split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                start, end = end, start
            selected.update(i for i in range(start, end + 1) if 0 <= i < num_layers)
        else:
            idx = int(token)
            if 0 <= idx < num_layers:
                selected.add(idx)
    return selected


def _rank_floor_target_layers(args, num_layers):
    scope = getattr(args, "truncate_min_rank_scope", "all_layers")
    if scope == "all_layers":
        return set(range(num_layers))
    if scope == "front_layers":
        if num_layers <= 2:
            return set(range(num_layers))
        end = max(1, num_layers // 3)
        return set(range(0, min(end, num_layers)))
    if scope == "middle_layers":
        if num_layers <= 2:
            return set(range(num_layers))
        start = num_layers // 3
        end = max(start + 1, (2 * num_layers) // 3)
        return set(range(start, min(end, num_layers)))
    if scope == "late_layers":
        if num_layers <= 2:
            return set(range(num_layers))
        start = max(0, (2 * num_layers) // 3)
        return set(range(start, num_layers))
    if scope == "custom":
        return _parse_rank_floor_layers(getattr(args, "truncate_min_rank_layers", None), num_layers)
    return set()


def _evenly_spaced_indices(limit, count):
    limit = int(limit)
    count = int(max(0, count))
    if limit <= 0 or count <= 0:
        return []
    if count >= limit:
        return list(range(limit))
    if count == 1:
        return [limit // 2]
    values = torch.linspace(0, limit - 1, steps=count)
    result = []
    seen = set()
    for value in values.tolist():
        idx = int(round(value))
        idx = max(0, min(limit - 1, idx))
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    for idx in range(limit):
        if len(result) >= count:
            break
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    return sorted(result)


def _empirical_fisher_target_layers(args, num_layers):
    explicit = getattr(args, "act_calib_debug_empirical_fisher_layers", None)
    if explicit:
        parsed = sorted(_parse_rank_floor_layers(explicit, num_layers))
        if parsed:
            return parsed, "explicit"
    target_count = int(max(0, getattr(args, "act_calib_debug_empirical_fisher_num_layers", 8)))
    return _evenly_spaced_indices(num_layers, target_count), "uniform_grid"


def _empirical_fisher_target_heads(args, num_heads):
    explicit = getattr(args, "act_calib_debug_empirical_fisher_heads", None)
    if explicit:
        parsed = sorted(_parse_rank_floor_layers(explicit, num_heads))
        if parsed:
            return parsed, "explicit"
    target_count = int(max(0, getattr(args, "act_calib_debug_empirical_fisher_num_heads", 8)))
    return _evenly_spaced_indices(num_heads, target_count), "uniform_grid"


def _empirical_fisher_enabled(args):
    return bool(
        getattr(args, "act_calib_debug", False)
        and getattr(args, "act_calib_debug_empirical_fisher", False)
        and getattr(args, "is_per_head_svd", False)
    )


def _empirical_fisher_module_config(args, module_name, layer_idx, proj_name, in_features, head_dim, selected_heads, device):
    in_proj_dim = min(
        int(max(1, getattr(args, "act_calib_debug_empirical_fisher_input_proj_dim", 8))),
        int(in_features),
    )
    out_proj_dim = min(
        int(max(1, getattr(args, "act_calib_debug_empirical_fisher_output_proj_dim", 8))),
        int(head_dim),
    )
    base_seed = int(getattr(args, "act_calib_debug_empirical_fisher_seed", 0))
    input_seed = debug_utils.stable_projection_seed(base_seed, f"{module_name}:input")
    input_projection = debug_utils.make_debug_projection(
        rows=int(in_features),
        cols=in_proj_dim,
        seed=input_seed,
        device=device,
    )
    config = {
        "selection_strategy": "uniform_layers_uniform_heads",
        "selected_layer_idx": int(layer_idx),
        "selected_head_indices": [int(h) for h in selected_heads],
        "head_dim": int(head_dim),
        "in_features": int(in_features),
        "input_proj_dim": int(in_proj_dim),
        "output_proj_dim": int(out_proj_dim),
        "input_projection": input_projection,
        "input_projection_seed": int(input_seed),
        "sample_count": 0,
        "sketch_dim": int(in_proj_dim * out_proj_dim),
        "heads": {},
    }
    for head_idx in selected_heads:
        output_seed = debug_utils.stable_projection_seed(base_seed, f"{module_name}:head:{int(head_idx)}:output")
        output_projection = debug_utils.make_debug_projection(
            rows=int(head_dim),
            cols=out_proj_dim,
            seed=output_seed,
            device=device,
        )
        sketch_dim = int(in_proj_dim * out_proj_dim)
        config["heads"][int(head_idx)] = {
            "sample_count": 0,
            "sketch_outer": torch.zeros(sketch_dim, sketch_dim, dtype=torch.float32),
            "sketch_diag": torch.zeros(sketch_dim, dtype=torch.float32),
            "sketch_trace_sum": 0.0,
            "output_projection": output_projection,
            "projection_seed": int(output_seed),
        }
    return config


def _enabled_proj_names(modules):
    enabled = {
        "q": {"q_proj"},
        "k": {"k_proj"},
        "v": {"v_proj"},
        "qv": {"q_proj", "v_proj"},
        "qkv": {"q_proj", "k_proj", "v_proj"},
    }.get(modules, {"q_proj", "k_proj", "v_proj"})
    return enabled


def _module_floor_enabled(args, proj_name):
    modules = str(getattr(args, "truncate_min_rank_modules", "qkv") or "qkv")
    enabled = _enabled_proj_names(modules)
    return proj_name in enabled


def _module_hybrid_enabled(args, proj_name):
    modules = str(getattr(args, "truncate_hybrid_modules", "qkv") or "qkv")
    enabled = _enabled_proj_names(modules)
    return proj_name in enabled


def _flatten_head_candidates(scores_cpu):
    candidates = []
    for head_idx in range(int(scores_cpu.shape[0])):
        for sv_idx in range(int(scores_cpu.shape[1])):
            score = float(scores_cpu[head_idx, sv_idx].item())
            candidates.append((score, head_idx, sv_idx))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _selection_count(selected_indices):
    if isinstance(selected_indices, dict):
        return sum(len(v) for v in selected_indices.values())
    return len(selected_indices)


def _selection_pairs(selection_map):
    pairs = set()
    for head_idx, indices in selection_map.items():
        for sv_idx in indices:
            pairs.add((int(head_idx), int(sv_idx)))
    return pairs


def _rebuild_selection_from_candidates(candidates, num_heads, total_selected, lambda_value):
    selected = {head_idx: [] for head_idx in range(num_heads)}
    taken = {head_idx: set() for head_idx in range(num_heads)}
    candidates_by_head = {head_idx: [] for head_idx in range(num_heads)}
    for score, head_idx, sv_idx in candidates:
        candidates_by_head[int(head_idx)].append((float(score), int(sv_idx)))

    cursor = {head_idx: 0 for head_idx in range(num_heads)}
    selected_total = 0
    while selected_total < total_selected:
        best_head = None
        best_adjusted = None
        best_sv = None
        for head_idx in range(num_heads):
            while cursor[head_idx] < len(candidates_by_head[head_idx]):
                score, sv_idx = candidates_by_head[head_idx][cursor[head_idx]]
                if sv_idx in taken[head_idx]:
                    cursor[head_idx] += 1
                    continue
                adjusted = score - lambda_value * len(selected[head_idx])
                if best_adjusted is None or adjusted > best_adjusted:
                    best_adjusted = adjusted
                    best_head = head_idx
                    best_sv = sv_idx
                break
        if best_head is None:
            break
        selected[best_head].append(best_sv)
        selected[best_head] = sorted(set(selected[best_head]))
        taken[best_head].add(best_sv)
        cursor[best_head] += 1
        selected_total += 1
    return selected


def apply_per_head_balance_penalty(layer_indices_dict, singular_values_dict, args, num_layers):
    if not bool(getattr(args, "truncate_balance_penalty", False)):
        return layer_indices_dict, {}

    lambda_value = float(getattr(args, "truncate_balance_lambda", 0.0) or 0.0)
    if lambda_value <= 0 or not getattr(args, "is_per_head_svd", False):
        return layer_indices_dict, {}

    target_layers = _rank_floor_target_layers(args, num_layers)
    balance_meta = {}
    grouped = {}
    for layer_idx, proj_map in layer_indices_dict.items():
        if layer_idx not in target_layers:
            continue
        layer_key = f"layer_{layer_idx}"
        for proj_name, selected_indices in list(proj_map.items()):
            if not _module_floor_enabled(args, proj_name):
                continue
            scores = singular_values_dict.get(layer_key, {}).get(proj_name)
            if not isinstance(scores, torch.Tensor) or scores.ndim < 2:
                continue
            scores_cpu = scores.detach().float().cpu()
            num_heads = int(scores_cpu.shape[0])
            selection_map = _normalize_selected_head_indices(selected_indices, num_heads)
            grouped.setdefault(proj_name, []).append(
                {
                    "layer_idx": int(layer_idx),
                    "module_name": f"model.layers.{layer_idx}.self_attn.{proj_name}",
                    "scores": scores_cpu,
                    "num_heads": num_heads,
                    "before_selection": selection_map,
                    "before_counts": [len(selection_map[h]) for h in range(num_heads)],
                    "total_selected": _selection_count(selection_map),
                }
            )

    for proj_name, modules in grouped.items():
        for mod in modules:
            candidates = _flatten_head_candidates(mod["scores"])
            new_selection = _rebuild_selection_from_candidates(
                candidates=candidates,
                num_heads=mod["num_heads"],
                total_selected=mod["total_selected"],
                lambda_value=lambda_value,
            )
            before_pairs = _selection_pairs(mod["before_selection"])
            after_pairs = _selection_pairs(new_selection)
            after_counts = [len(new_selection[h]) for h in range(mod["num_heads"])]
            layer_indices_dict[mod["layer_idx"]][proj_name] = {
                head_idx: sorted(new_selection[head_idx]) for head_idx in range(mod["num_heads"])
            }
            balance_meta[mod["module_name"]] = {
                "floor_enabled": False,
                "floor_target": None,
                "floor_before_head_count": mod["before_counts"],
                "floor_after_head_count": after_counts,
                "floor_added_count": len(after_pairs - before_pairs),
                "floor_removed_count": len(before_pairs - after_pairs),
                "floor_unsatisfied": False,
                "floor_cross_layer_enabled": False,
                "balance_penalty_enabled": True,
                "balance_penalty_lambda": lambda_value,
            }

    return layer_indices_dict, balance_meta


def _layer_budget_counts_from_layer_indices(layer_indices_dict, args=None, enabled_fn=None):
    budget_counts = {}
    for layer_idx, proj_map in layer_indices_dict.items():
        layer_budget = budget_counts.setdefault(int(layer_idx), {})
        for proj_name, selected_indices in proj_map.items():
            if enabled_fn is not None and not enabled_fn(args, proj_name):
                continue
            layer_budget[proj_name] = _selection_count(selected_indices)
    return budget_counts


def _select_top_within_layer(scores_cpu, total_count):
    if total_count <= 0:
        return {}
    candidates = _flatten_head_candidates(scores_cpu)
    selected = {head_idx: [] for head_idx in range(int(scores_cpu.shape[0]))}
    for _, head_idx, sv_idx in candidates[:total_count]:
        selected[int(head_idx)].append(int(sv_idx))
    return {head_idx: sorted(values) for head_idx, values in selected.items()}


def apply_layer_budget_counts_hybrid(layer_indices_dict, singular_values_dict, target_budget_counts, args, num_layers, meta_flag_name):
    hybrid_meta = {}
    target_layers = set(range(num_layers))
    proj_totals_before = {}
    proj_totals_after = {}
    for layer_idx in target_layers:
        layer_key = f"layer_{layer_idx}"
        proj_map = layer_indices_dict.get(layer_idx, {})
        target_proj_map = target_budget_counts.get(layer_idx, {})
        for proj_name, selected_indices in list(proj_map.items()):
            if not _module_hybrid_enabled(args, proj_name):
                continue
            if proj_name not in target_proj_map:
                continue
            scores = singular_values_dict.get(layer_key, {}).get(proj_name)
            if not isinstance(scores, torch.Tensor) or scores.ndim < 2:
                continue
            scores_cpu = scores.detach().float().cpu()
            target_count = int(target_proj_map.get(proj_name, _selection_count(selected_indices)))
            target_count = max(0, min(target_count, scores_cpu.shape[0] * scores_cpu.shape[1]))
            before_map = _normalize_selected_head_indices(selected_indices, int(scores_cpu.shape[0]))
            after_map = _select_top_within_layer(scores_cpu, target_count)
            before_pairs = _selection_pairs(before_map)
            after_pairs = _selection_pairs(after_map)
            before_total = len(before_pairs)
            after_total = len(after_pairs)
            proj_totals_before[proj_name] = proj_totals_before.get(proj_name, 0) + before_total
            proj_totals_after[proj_name] = proj_totals_after.get(proj_name, 0) + after_total
            layer_indices_dict[layer_idx][proj_name] = after_map
            if before_total != after_total:
                logging.info(f"layer {layer_idx} {proj_name} before={before_total} after={after_total}")
            hybrid_meta[f"model.layers.{layer_idx}.self_attn.{proj_name}"] = {
                "floor_enabled": False,
                "floor_target": None,
                "floor_before_head_count": [len(before_map[h]) for h in range(int(scores_cpu.shape[0]))],
                "floor_after_head_count": [len(after_map[h]) for h in range(int(scores_cpu.shape[0]))],
                "floor_added_count": len(after_pairs - before_pairs),
                "floor_removed_count": len(before_pairs - after_pairs),
                "floor_unsatisfied": False,
                "floor_cross_layer_enabled": False,
                meta_flag_name: True,
            }
    for proj_name in sorted(proj_totals_before):
        logging.info(
            f"proj={proj_name} total_before={proj_totals_before.get(proj_name, 0)} "
            f"total_after={proj_totals_after.get(proj_name, 0)}"
        )
    return layer_indices_dict, hybrid_meta


def build_loss_aware_layer_budget_counts(layer_indices_dict, singular_values_dict, args, num_layers):
    budget_counts = {}
    for layer_idx in range(num_layers):
        layer_key = f"layer_{layer_idx}"
        proj_map = layer_indices_dict.get(layer_idx, {})
        if not proj_map:
            continue
        for proj_name, selected_indices in proj_map.items():
            if not _module_hybrid_enabled(args, proj_name):
                continue
            scores = singular_values_dict.get(layer_key, {}).get(proj_name)
            if not isinstance(scores, torch.Tensor) or scores.ndim < 2:
                continue
            score_cpu = scores.detach().float().cpu()
            budget_counts.setdefault(proj_name, []).append(
                {
                    "layer_idx": int(layer_idx),
                    "energy": float((score_cpu ** 2).sum().item()),
                    "count": int(_selection_count(selected_indices)),
                    "capacity": int(score_cpu.shape[0] * score_cpu.shape[1]),
                }
            )

    target_budget_counts = {}
    for proj_name, entries in budget_counts.items():
        total_budget_proj = sum(entry["count"] for entry in entries)
        total_energy = sum(entry["energy"] for entry in entries)
        if total_budget_proj <= 0:
            continue
        if total_energy <= 0:
            for entry in entries:
                target_budget_counts.setdefault(entry["layer_idx"], {})[proj_name] = entry["count"]
            continue

        entries_by_layer = {entry["layer_idx"]: entry for entry in entries}
        remainders = []
        assigned_total = 0
        for entry in entries:
            raw_count = total_budget_proj * entry["energy"] / total_energy
            base_count = min(int(math.floor(raw_count)), entry["capacity"])
            target_budget_counts.setdefault(entry["layer_idx"], {})[proj_name] = base_count
            remainders.append((raw_count - base_count, entry["layer_idx"]))
            assigned_total += base_count

        remaining = total_budget_proj - assigned_total
        if remaining > 0:
            sorted_remainders = sorted(remainders, key=lambda item: (-item[0], item[1]))
            while remaining > 0:
                progress = False
                for _, layer_idx in sorted_remainders:
                    current_count = target_budget_counts[layer_idx][proj_name]
                    capacity = entries_by_layer[layer_idx]["capacity"]
                    if current_count >= capacity:
                        continue
                    target_budget_counts[layer_idx][proj_name] += 1
                    remaining -= 1
                    progress = True
                    if remaining == 0:
                        break
                if not progress:
                    logging.warning(
                        f"Unable to fully preserve total budget for {proj_name}: "
                        f"remaining={remaining}, total_budget={total_budget_proj}"
                    )
                    break

    return target_budget_counts


def apply_loss_aware_layer_budget_hybrid(layer_indices_dict, singular_values_dict, args, num_layers):
    if not bool(getattr(args, "truncate_hybrid_loss_aware_layer_budget", False)):
        return layer_indices_dict, {}
    logging.info(
        "Applying loss-aware layer-budget hybrid for modules=%s",
        str(getattr(args, "truncate_hybrid_modules", "qkv") or "qkv"),
    )
    target_budget_counts = build_loss_aware_layer_budget_counts(
        layer_indices_dict=layer_indices_dict,
        singular_values_dict=singular_values_dict,
        args=args,
        num_layers=num_layers,
    )
    return apply_layer_budget_counts_hybrid(
        layer_indices_dict=layer_indices_dict,
        singular_values_dict=singular_values_dict,
        target_budget_counts=target_budget_counts,
        args=args,
        num_layers=num_layers,
        meta_flag_name="hybrid_loss_aware_layer_budget",
    )


def apply_grad_layer_budget_hybrid(layer_indices_dict, singular_values_dict, grad_layer_indices_dict, args, num_layers):
    if not bool(getattr(args, "truncate_hybrid_grad_layer_budget", False)):
        return layer_indices_dict, {}
    target_budget_counts = _layer_budget_counts_from_layer_indices(
        grad_layer_indices_dict,
        args=args,
        enabled_fn=_module_hybrid_enabled,
    )
    return apply_layer_budget_counts_hybrid(
        layer_indices_dict=layer_indices_dict,
        singular_values_dict=singular_values_dict,
        target_budget_counts=target_budget_counts,
        args=args,
        num_layers=num_layers,
        meta_flag_name="hybrid_grad_layer_budget",
    )


def _normalize_selected_head_indices(selected_indices, num_heads):
    normalized = {head_idx: [] for head_idx in range(num_heads)}
    if isinstance(selected_indices, dict):
        for head_idx in range(num_heads):
            values = selected_indices.get(head_idx, selected_indices.get(str(head_idx), []))
            if values is None:
                values = []
            normalized[head_idx] = sorted({int(v) for v in values})
    else:
        for head_idx in range(num_heads):
            values = selected_indices[head_idx] if head_idx < len(selected_indices) else []
            normalized[head_idx] = sorted({int(v) for v in values})
    return normalized


def _select_best_unpicked(score_row, selected_set):
    ranked = torch.argsort(score_row.detach().float().cpu(), descending=True).tolist()
    for idx in ranked:
        if int(idx) not in selected_set:
            return int(idx)
    return None


def _proj_priority(proj_name):
    if proj_name == "v_proj":
        return 0
    if proj_name in {"q_proj", "k_proj"}:
        return 1
    return 2


def _receiver_sort_key(item, proj_name):
    layer_idx, head_idx, deficit, current_count = item
    return (-deficit, layer_idx, _proj_priority(proj_name), head_idx)


def _select_cross_layer_donor(receiver_layer_idx, candidates, floor):
    viable = []
    for donor in candidates:
        if donor["count"] <= floor:
            continue
        viable.append(donor)
    if not viable:
        return None
    viable.sort(
        key=lambda d: (
            -(d["count"] - floor),
            d["score"],
            -d["layer_idx"],
            d["head_idx"],
            d["sv_idx"],
        )
    )
    return viable[0]


def apply_per_head_rank_floor(layer_indices_dict, singular_values_dict, args, num_layers):
    floor = int(getattr(args, "truncate_min_rank_per_head", 4) or 0)
    if floor <= 0 or not getattr(args, "is_per_head_svd", False):
        return layer_indices_dict, {}

    strict = bool(getattr(args, "truncate_min_rank_strict", True))
    target_layers = _rank_floor_target_layers(args, num_layers)

    floor_meta = {}

    grouped = {}
    for layer_idx, proj_map in layer_indices_dict.items():
        if layer_idx not in target_layers:
            continue
        layer_key = f"layer_{layer_idx}"
        for proj_name, selected_indices in list(proj_map.items()):
            if not _module_floor_enabled(args, proj_name):
                continue
            scores = singular_values_dict.get(layer_key, {}).get(proj_name)
            if not isinstance(scores, torch.Tensor) or scores.ndim < 2:
                continue
            score_cpu = scores.detach().float().cpu()
            num_heads = int(score_cpu.shape[0])
            selection_map = _normalize_selected_head_indices(selected_indices, num_heads)
            module_name = f"model.layers.{layer_idx}.self_attn.{proj_name}"
            grouped.setdefault(proj_name, []).append(
                {
                    "layer_idx": int(layer_idx),
                    "module_name": module_name,
                    "scores": score_cpu,
                    "selection_map": selection_map,
                    "num_heads": num_heads,
                    "before_counts": [len(selection_map[h]) for h in range(num_heads)],
                }
            )

    for proj_name, modules in grouped.items():
        total_selected = sum(sum(len(v) for v in mod["selection_map"].values()) for mod in modules)
        total_heads = sum(mod["num_heads"] for mod in modules)
        required_total = total_heads * floor
        if strict and total_selected < required_total:
            layer_list = sorted(mod["layer_idx"] for mod in modules)
            raise ValueError(
                f"Strict per-head rank floor cannot be satisfied for {proj_name}: "
                f"selected_count_total={total_selected}, required_min={required_total}, "
                f"layers={layer_list}, floor={floor}"
            )

        for mod in modules:
            mod["added_count"] = 0
            mod["removed_count"] = 0

        receivers = []
        for mod in modules:
            for head_idx in range(mod["num_heads"]):
                current_count = len(mod["selection_map"][head_idx])
                if current_count < floor:
                    receivers.append((mod["layer_idx"], head_idx, floor - current_count, current_count, mod))
        receivers.sort(key=lambda item: _receiver_sort_key(item[:4], proj_name))

        for receiver_layer_idx, recv_head, _, _, recv_mod in receivers:
            while len(recv_mod["selection_map"][recv_head]) < floor:
                recv_selected = set(recv_mod["selection_map"][recv_head])
                best_idx = _select_best_unpicked(recv_mod["scores"][recv_head], recv_selected)
                if best_idx is None:
                    break
                donor_candidates = []
                for donor_mod in modules:
                    for donor_head in range(donor_mod["num_heads"]):
                        donor_count = len(donor_mod["selection_map"][donor_head])
                        if donor_count <= floor:
                            continue
                        for sv_idx in donor_mod["selection_map"][donor_head]:
                            score = float(donor_mod["scores"][donor_head, sv_idx].item())
                            donor_candidates.append(
                                {
                                    "layer_idx": donor_mod["layer_idx"],
                                    "head_idx": donor_head,
                                    "sv_idx": int(sv_idx),
                                    "score": score,
                                    "count": donor_count,
                                    "module": donor_mod,
                                }
                            )
                donor = _select_cross_layer_donor(receiver_layer_idx, donor_candidates, floor) if strict else None
                if strict and donor is None:
                    layer_list = sorted(mod["layer_idx"] for mod in modules)
                    raise ValueError(
                        f"Strict per-head rank floor donor search failed for {proj_name}: "
                        f"receiver_layer={receiver_layer_idx}, receiver_head={recv_head}, "
                        f"layers={layer_list}, floor={floor}"
                    )

                recv_mod["selection_map"][recv_head].append(best_idx)
                recv_mod["selection_map"][recv_head] = sorted(set(recv_mod["selection_map"][recv_head]))
                recv_mod["added_count"] += 1

                if donor is not None:
                    donor_mod = donor["module"]
                    donor_head = donor["head_idx"]
                    donor_idx = donor["sv_idx"]
                    donor_mod["selection_map"][donor_head].remove(donor_idx)
                    donor_mod["removed_count"] += 1

        for mod in modules:
            after_counts = [len(mod["selection_map"][h]) for h in range(mod["num_heads"])]
            floor_unsatisfied = any(count < floor for count in after_counts)
            if strict and floor_unsatisfied:
                raise ValueError(
                    f"Strict per-head rank floor left unsatisfied heads for {proj_name}: "
                    f"layer={mod['layer_idx']}, before={mod['before_counts']}, after={after_counts}, floor={floor}"
                )
            layer_indices_dict[mod["layer_idx"]][proj_name] = {
                head_idx: sorted(mod["selection_map"][head_idx]) for head_idx in range(mod["num_heads"])
            }
            floor_meta[mod["module_name"]] = {
                "floor_enabled": True,
                "floor_target": floor,
                "floor_before_head_count": mod["before_counts"],
                "floor_after_head_count": after_counts,
                "floor_added_count": mod["added_count"],
                "floor_removed_count": mod["removed_count"],
                "floor_unsatisfied": floor_unsatisfied,
                "floor_cross_layer_enabled": True,
            }

    return layer_indices_dict, floor_meta


def rebuild_topk_from_layer_indices(layer_indices_dict, singular_values_dict, is_per_head_svd=False):
    rebuilt_indices = {}
    rebuilt_scores = {}
    collected = {}
    for layer_idx, proj_map in layer_indices_dict.items():
        layer_key = f"layer_{layer_idx}"
        for proj_name, selected_indices in proj_map.items():
            scores = singular_values_dict.get(layer_key, {}).get(proj_name)
            if not isinstance(scores, torch.Tensor):
                continue
            if is_per_head_svd and isinstance(selected_indices, dict) and scores.ndim >= 2:
                items = collected.setdefault(proj_name, [])
                for head_idx, idxs in selected_indices.items():
                    for sv_idx in idxs:
                        score = float(scores[int(head_idx), int(sv_idx)].detach().float().cpu().item())
                        items.append((score, int(layer_idx), int(sv_idx), int(head_idx)))
            else:
                items = collected.setdefault(proj_name, [])
                for sv_idx in selected_indices:
                    score = float(scores[int(sv_idx)].detach().float().cpu().item())
                    items.append((score, int(layer_idx), int(sv_idx)))
    for proj_name, items in collected.items():
        items.sort(key=lambda x: x[0], reverse=True)
        if is_per_head_svd and len(items) > 0 and len(items[0]) == 4:
            rebuilt_indices[proj_name] = [(layer, sv_idx, head_idx) for _, layer, sv_idx, head_idx in items]
            rebuilt_scores[proj_name] = [score for score, _, _, _ in items]
        else:
            rebuilt_indices[proj_name] = [(layer, sv_idx) for _, layer, sv_idx in items]
            rebuilt_scores[proj_name] = [score for score, _, _ in items]
    return rebuilt_indices, rebuilt_scores


def _selected_count(selected_indices):
    if isinstance(selected_indices, dict):
        return int(sum(len(v) for v in selected_indices.values()))
    return int(len(selected_indices or []))


def _selection_budget_snapshot(layer_indices_dict):
    snapshot = {}
    for layer_idx, proj_map in layer_indices_dict.items():
        layer_entry = snapshot.setdefault(int(layer_idx), {})
        for proj_name, selected_indices in proj_map.items():
            if proj_name not in {"q_proj", "k_proj", "v_proj"}:
                continue
            if isinstance(selected_indices, dict):
                max_head = max((int(h) for h in selected_indices.keys()), default=-1)
                head_budget = [0] * (max_head + 1 if max_head >= 0 else 0)
                for head_idx, idxs in selected_indices.items():
                    head_budget[int(head_idx)] = len(list(idxs))
            else:
                head_budget = []
            layer_entry[proj_name] = {
                "layer_budget": _selected_count(selected_indices),
                "head_budget": head_budget,
                "total_budget": _selected_count(selected_indices),
            }
    return snapshot

def _act_calib_cache_tag(args):
    enable = (
        bool(getattr(args, "act_calib_fake_quant", False))
    )
    if not enable:
        return ""

    a_bits = int(getattr(args, "a_bits_calib", 16))
    a_groupsize = int(getattr(args, "a_groupsize", -1))
    a_asym = bool(getattr(args, "a_asym", False))
    a_clip_ratio = float(getattr(args, "a_clip_ratio", 1.0))
    clip_tag = str(a_clip_ratio).replace(".", "p")
    sym_tag = "asym" if a_asym else "sym"
    return f"_afq_ab{a_bits}_ag{a_groupsize}_{sym_tag}_ac{clip_tag}"

@torch.enable_grad()
def calib_grad_distribution(model, dataloader, tokenizer, image_processor, args, use_cache=True, cache_file=None):
    """
    Calculate Grad distribution for each layer's output
    
    Args:
        model: Model to be calibrated
        tokenizer: Tokenizer
        image_processor: Image processor
        args: Parameter configuration
        use_cache: Whether to use cache
        cache_file: Cache file path, automatically generated if None
    """
    model_id = model.config._name_or_path
    
    if cache_file is None:
        cache_dir  = args.act_cache_dir + "/cache"
        if args.cache_in_log:
            cache_dir = args.act_cache_dir + "/cache"
        os.makedirs(cache_dir, exist_ok=True)
        # Add relevant information to cache file name
        if args.qkv_fuse:
            raise ValueError("QKV fuse is not supported for output gradient distribution")
        elif args.kv_fuse:
            raise ValueError("KV fuse is not supported for output gradient distribution")
        else:
            fusion_mode = "separate"
        if bool(getattr(args, "pre_svd_rotate", False)):
            rotate_info = "presvdrot"
        else:
            rotate_info = "presvdnorot"
            
        method = args.calib_method if hasattr(args, "act_aware") and args.act_aware else "no_act_aware"
        if method != "cholesky":
            raise ValueError("Only cholesky calibration method is supported for output gradient distribution")
        
        if args.misalign_text_image:
            fusion_mode = fusion_mode + "_misalign"
        if args.is_per_head_svd:
            fusion_mode = fusion_mode + "_perhead"
        if args.taylor_order != 2:
            fusion_mode = fusion_mode + "_taylor" + str(args.taylor_order)
        calib_tag = _act_calib_cache_tag(args) + reasoning_calib_utils.cache_tag(
            args,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
        )
        if args.a_clip_ratio == 1.0:
            cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_{args.nsamples}_{args.seed}_{rotate_info}_{args.svd_modules}_{fusion_mode}{calib_tag}_calib_output_grad_distribution_{method}_args{args.svd_modules}.pt")
        else:
            cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_aclip{args.a_clip_ratio}_{args.nsamples}_{args.seed}_{rotate_info}_{args.svd_modules}_{fusion_mode}{calib_tag}_calib_output_grad_distribution_{method}_args{args.svd_modules}.pt")
    
    cache_file = calib_cache.released_file(args, "output_grad") or cache_file
    if os.path.exists(cache_file) and use_cache:
        logging.info(f"Loading output gradient distribution cache from {cache_file}...")
        all_output_grad_scaling_matrix = torch.load(cache_file, map_location="cpu")
        # Load output gradient distribution into the self_attn.S_grad_info attribute of corresponding layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'lm_head' not in name and _is_lm_layer_name(name):
                if name in all_output_grad_scaling_matrix:
                    if args.svd_modules in ['qkv', 'attn', 'all']:
                        if args.qkv_fuse:
                            raise ValueError("QKV fuse is not supported for output gradient distribution")
                        elif args.kv_fuse:
                            raise ValueError("KV fuse is not supported for output gradient distribution")
                        else:
                            # for separate SVD, output gradient distribution is on q_proj, k_proj, v_proj, o_proj
                            module.output_grad_scaling_matrix = all_output_grad_scaling_matrix[name].to(utils.get_dev())
                    else:
                        # for separate SVD, output gradient distribution is on mlp.up_proj, mlp.gate_proj, mlp.down_proj
                        module.output_grad_scaling_matrix = all_output_grad_scaling_matrix[name].to(utils.get_dev())
        logging.info("Successfully loaded output gradient distribution cache!")
        if getattr(args, "act_calib_debug", False):
            logging.info(
                "act_calib_debug: cache hit for output-gradient calibration — raw covariance tensors were not recomputed; "
                "run with --no-use_cache or use_cache=False for online grad debug dump."
            )
        return

    print("Starting output gradient distribution calculation...")
    logging.info('start output gradient distribution computing')
    model.eval()

    # Ensure the entire model is on CUDA
    device = utils.get_dev()
    model = model.to(device)
    reasoning_batches = None
    if reasoning_calib_utils.enabled(args):
        reasoning_batches = reasoning_calib_utils.materialize_reasoning_batches(
            model=model,
            dataloader=dataloader,
            tokenizer=tokenizer,
            image_processor=image_processor,
            args=args,
            device=device,
        )

    # Set model to training mode and only allow gradient computation for svd_modules layers
    model.train()
    
    # only collect output gradient, not weight gradient
    # thus we can set weight gradient only for layer 0's q_proj, k_proj, v_proj
    # 先关掉所有参数梯度
    for _, p in model.named_parameters():
        p.requires_grad = False

    # 按索引开启需要梯度的少量层（比如只开第 0 层）
    layers = model_utils.get_layers(model)
    num_layers = len(layers)
    attn = layers[0].self_attn
    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
        if hasattr(attn, proj_name):
            for p in getattr(attn, proj_name).parameters():
                p.requires_grad = True

    handles = {}
    module_name_lookup = {id(module): name for name, module in model.named_modules()}
    num_heads_q = layers[0].self_attn.config.num_attention_heads #model.config.num_attention_heads
    num_heads_kv = layers[0].self_attn.config.num_key_value_heads
    head_dim = layers[0].self_attn.config.hidden_size // layers[0].self_attn.config.num_attention_heads

    hook_trigger_count = {}  # debug: count backward hook invocations per key
    grad_debug_payload = {} if getattr(args, "act_calib_debug", False) else None
    empirical_fisher_active = _empirical_fisher_enabled(args)
    empirical_target_layers, empirical_layer_strategy = _empirical_fisher_target_layers(args, num_layers)
    empirical_target_layers = set(empirical_target_layers)
    empirical_forward_handles = {}

    def make_output_grad_hook(name):
        def hook(module, grad_in, grad_out):
            hook_trigger_count[name] = hook_trigger_count.get(name, 0) + 1
            base_grad = grad_out[0].detach()
            token_count = int(base_grad.numel() // max(int(base_grad.shape[-1]), 1))
            sample_count = int(base_grad.shape[0] if base_grad.dim() > 1 else 1)
            module._debug_hook_count = int(getattr(module, "_debug_hook_count", 0) + 1)
            module._debug_sample_count = int(getattr(module, "_debug_sample_count", 0) + sample_count)
            module._debug_token_count = int(getattr(module, "_debug_token_count", 0) + token_count)
            bsz = grad_out[0].shape[0]
            if args.is_per_head_svd and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name):
                # [batch_size, num_heads, seq_len, head_dim]
                if 'q_proj' in name:
                    grads = grad_out[0].detach().view(bsz, -1, num_heads_q, head_dim).permute(0, 2, 1, 3).contiguous().float()
                elif 'k_proj' in name:
                    grads = grad_out[0].detach().view(bsz, -1, num_heads_kv, head_dim).permute(0, 2, 1, 3).contiguous().float()
                elif 'v_proj' in name:
                    grads = grad_out[0].detach().view(bsz, -1, num_heads_kv, head_dim).permute(0, 2, 1, 3).contiguous().float()
                else:
                    raise ValueError(f"Invalid projection name: {name}")
            else:
                # [batch_size, seq_len, hidden_size]
                grads = grad_out[0].detach().float()
            # calculate covariance of output gradients
            grads_cov = torch.matmul(grads.transpose(-1, -2), grads).sum(dim=0) # [hidden_size, hidden_size] or [num_heads, head_dim, head_dim]
            if not hasattr(module, "output_grad_distribution"):
                module.register_buffer("output_grad_distribution", torch.zeros_like(grads_cov, dtype=torch.float32))
            module.output_grad_distribution += grads_cov
            empirical_cfg = getattr(module, "_empirical_fisher_cfg", None)
            input_proj = getattr(module, "_empirical_fisher_input_proj", None)
            if empirical_cfg is not None and isinstance(input_proj, torch.Tensor) and input_proj.numel() > 0:
                for head_idx, head_state in empirical_cfg["heads"].items():
                    if grads.ndim < 4 or int(head_idx) >= int(grads.shape[1]):
                        continue
                    output_projection = head_state["output_projection"].to(device=grads.device, dtype=grads.dtype)
                    grad_head = grads[:, int(head_idx), :, :]
                    grad_proj = torch.einsum("bth,hk->btk", grad_head, output_projection)
                    sketch = torch.einsum("btk,btr->bkr", grad_proj, input_proj)
                    flat = sketch.reshape(sketch.shape[0], -1).detach().float().cpu()
                    head_state["sketch_outer"] += flat.transpose(0, 1) @ flat
                    head_state["sketch_diag"] += torch.sum(flat * flat, dim=0)
                    head_state["sketch_trace_sum"] += float(torch.sum(flat * flat).item())
                    head_state["sample_count"] += int(flat.shape[0])
                    empirical_cfg["sample_count"] += int(flat.shape[0])
                delattr(module, "_empirical_fisher_input_proj")
            del grads, grads_cov
        return hook

    def make_empirical_input_hook(module_name):
        def hook(module, inputs, output):
            empirical_cfg = getattr(module, "_empirical_fisher_cfg", None)
            if empirical_cfg is None:
                return
            if not inputs:
                return
            inp = inputs[0].detach()
            if not isinstance(inp, torch.Tensor):
                return
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            inp = inp.float()
            input_projection = empirical_cfg["input_projection"].to(device=inp.device, dtype=inp.dtype)
            module._empirical_fisher_input_proj = torch.einsum("btd,dk->btk", inp, input_projection).detach()
        return hook
    
    # Register backward hooks for q/k/v on selected layer indices
    # args.grad_hook_layer_idx can be int, list/tuple of int, or comma-separated string, e.g., "0,3,5"
    
    if hasattr(args, 'grad_hook_layer_idx'):
        raw_idx = args.grad_hook_layer_idx
        if isinstance(raw_idx, (list, tuple)):
            target_indices = [int(i) for i in raw_idx]
        else:
            raw_str = str(raw_idx).strip()
            target_indices = [int(x) for x in raw_str.split(',') if x != '']
    else:
        target_indices = list(range(num_layers))

    if len(target_indices) > 0:
        for idx, layer in enumerate(model_utils.get_layers(model)):
            if idx not in target_indices:
                continue
            if args.svd_modules in ['qkv', 'attn', 'all']:
                for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                    if hasattr(layer.self_attn, proj_name):
                        proj = getattr(layer.self_attn, proj_name)
                        key = f"layer_{idx}.{proj_name}"
                        handles[key] = proj.register_full_backward_hook(make_output_grad_hook(key))
                        proj.output_grad_distribution = 0
                        hook_trigger_count[key] = 0
                        if empirical_fisher_active and idx in empirical_target_layers:
                            num_proj_heads = num_heads_q if proj_name == 'q_proj' else num_heads_kv
                            selected_heads, head_strategy = _empirical_fisher_target_heads(args, num_proj_heads)
                            if selected_heads:
                                module_name = module_name_lookup.get(id(proj), f"model.layers.{idx}.self_attn.{proj_name}")
                                proj._empirical_fisher_cfg = _empirical_fisher_module_config(
                                    args=args,
                                    module_name=module_name,
                                    layer_idx=idx,
                                    proj_name=proj_name,
                                    in_features=proj.in_features,
                                    head_dim=head_dim,
                                    selected_heads=selected_heads,
                                    device=proj.weight.device,
                                )
                                proj._empirical_fisher_cfg["selection_strategy"] = f"{empirical_layer_strategy}_{head_strategy}"
                                empirical_forward_handles[key] = proj.register_forward_hook(make_empirical_input_hook(module_name))
            if args.svd_modules in ['o_proj', 'attn', 'all']:
                if hasattr(layer.self_attn, 'o_proj'):
                    proj = layer.self_attn.o_proj
                    key = f"layer_{idx}.o_proj"
                    handles[key] = proj.register_full_backward_hook(make_output_grad_hook(key))
                    proj.output_grad_distribution = 0
                    hook_trigger_count[key] = 0
            if args.svd_modules in ['gaup', 'mlp', 'all']:
                for proj_name in ['up_proj', 'gate_proj']:
                    if hasattr(layer.mlp, proj_name):
                        proj = getattr(layer.mlp, proj_name)
                        key = f"layer_{idx}.{proj_name}"
                        handles[key] = proj.register_full_backward_hook(make_output_grad_hook(key))
                        proj.output_grad_distribution = 0
                        hook_trigger_count[key] = 0
            if args.svd_modules in ['down', 'mlp', 'all']:
                if hasattr(layer.mlp, 'down_proj'):
                    proj = layer.mlp.down_proj
                    key = f"layer_{idx}.down_proj"
                    handles[key] = proj.register_full_backward_hook(make_output_grad_hook(key))
                    proj.output_grad_distribution = 0
                    hook_trigger_count[key] = 0
    
    accumulation_steps = 1   # Number of accumulated batches
    batch_count = 0          # Accumulated batch counter
    
    grad_iter = reasoning_batches if reasoning_batches is not None else dataloader
    for batch_idx, batch in enumerate(tqdm(grad_iter, desc="Computing Output Gradient Distributions")):
        try:
            is_qwen_model = grad_info_utils._is_qwen_model(args, model, tokenizer)
            if reasoning_batches is not None:
                reasoning_batch = batch
                with torch.enable_grad():
                    outputs = reasoning_calib_utils.forward_reasoning_batch(model, reasoning_batch)
                    loss = reasoning_calib_utils.compute_reasoning_loss(outputs, reasoning_batch, args)
                    loss = loss / accumulation_steps
                    loss.backward()
            elif tokenizer is None and image_processor is None:
                input_ids = batch[0].to(device)
                labels = input_ids.clone()
                if args.label_shift: # enable label shift like palu, or spinquant handle this in Trainer?
                    labels = labels[:, 1:]
                    input_ids = input_ids[:, :-1]
                with torch.enable_grad():
                    outputs = model(input_ids, labels=labels)
                    loss = outputs[0]
                    loss = loss / accumulation_steps  # Normalize loss
                    loss.backward()
            elif is_qwen_model:
                # Qwen2/VL, Qwen3VL: batch is list of message dicts, use train inputs with labels
                inputs, labels = grad_info_utils._build_qwen_train_inputs(batch, image_processor, device, args=args)
                with torch.enable_grad():
                    outputs = model(**inputs, labels=labels)
                    loss = outputs[0]
                    loss = loss / accumulation_steps
                    loss.backward()
            elif molmo_utils.is_molmo_model(args=args, model=model, tokenizer=tokenizer):
                inputs, labels = molmo_utils.build_molmo_train_inputs(
                    batch,
                    image_processor,
                    device,
                    model=model,
                    label_mode=args.label_mode,
                    label_shift=bool(getattr(args, "label_shift", True)),
                )
                with torch.enable_grad():
                    outputs = model(**inputs, labels=labels)
                    loss = outputs[0]
                    loss = loss / accumulation_steps
                    loss.backward()
            elif tokenizer is None: # SmolVLM
                    # Use message_to_prompt_train to process batch data
                inputs, _, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                # Define recursive function to move nested tensor structures to specified device
                def move_to_device(obj, target_device):
                    if isinstance(obj, torch.Tensor):
                        return obj.to(target_device)
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(move_to_device(item, target_device) for item in obj)
                    elif isinstance(obj, dict):
                        return {k: move_to_device(v, target_device) for k, v in obj.items()}
                    else:
                        return obj
                
                # Move data to corresponding device
                inputs = move_to_device(inputs, device)
                output_ids = move_to_device(output_ids, device)

                input_ids = inputs.get('input_ids')
                if not isinstance(output_ids, torch.Tensor) and hasattr(output_ids, 'get'):
                    output_ids = output_ids.get('input_ids', output_ids)
                if isinstance(output_ids, dict):
                    output_ids = output_ids.get('input_ids')
                
                # Adjust input and label lengths to match
                # breakpoint()
                if input_ids.size(1) != output_ids.size(1):
                    max_len = max(input_ids.size(1), output_ids.size(1))
                    if input_ids.size(1) < max_len:
                        padding = torch.zeros((input_ids.size(0), max_len - input_ids.size(1)), 
                                            dtype=input_ids.dtype, device=input_ids.device)
                        input_ids = torch.cat([input_ids, padding], dim=1)
                    else:
                        input_ids = input_ids[:, :max_len]
                    if output_ids.size(1) < max_len:
                        padding = torch.full((output_ids.size(0), max_len - output_ids.size(1)), 
                                            fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
                        output_ids = torch.cat([output_ids, padding], dim=1)
                    else:
                        output_ids = output_ids[:, :max_len]
                    print(f"Adjusted input and label lengths to {max_len}")
                    input_ids = input_ids.to(device)
                    output_ids = output_ids.to(device)
                
                inputs['input_ids'] = input_ids
                inputs['attention_mask'] = input_ids.ne(0).to(device)


                # Calculate loss for current batch and perform gradient accumulation
                with torch.enable_grad():
                    outputs = model(**inputs, labels=output_ids)
                    loss = outputs[0]
                    loss = loss / accumulation_steps  # Normalize loss
                    loss.backward()
            elif tokenizer == 'hf_v16': # LLaVA Next/v1.6
                    # Use message_to_prompt_train to process batch data
                inputs, _, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                
                # Define recursive function to move nested tensor structures to specified device
                def move_to_device(obj, target_device):
                    if isinstance(obj, torch.Tensor):
                        return obj.to(target_device)
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(move_to_device(item, target_device) for item in obj)
                    elif isinstance(obj, dict):
                        return {k: move_to_device(v, target_device) for k, v in obj.items()}
                    else:
                        return obj
                
                # Move data to corresponding device
                inputs = move_to_device(inputs, device)
                output_ids = move_to_device(output_ids, device)
                input_ids = inputs.get('input_ids')
                if not isinstance(output_ids, torch.Tensor) and hasattr(output_ids, 'get'):
                    output_ids = output_ids.get('input_ids', output_ids)
                if isinstance(output_ids, dict):
                    output_ids = output_ids.get('input_ids')

                # breakpoint()
                # 0 as pad token id, following many tokenizers
                if input_ids.size(1) != output_ids.size(1):
                    max_len = max(input_ids.size(1), output_ids.size(1))
                    if input_ids.size(1) < max_len:
                        padding = torch.zeros((input_ids.size(0), max_len - input_ids.size(1)), 
                                            dtype=input_ids.dtype, device=input_ids.device)
                        input_ids = torch.cat([input_ids, padding], dim=1)
                    else:
                        input_ids = input_ids[:, :max_len]
                    if output_ids.size(1) < max_len:
                        padding = torch.full((output_ids.size(0), max_len - output_ids.size(1)), 
                                            fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
                        output_ids = torch.cat([output_ids, padding], dim=1)
                    else:
                        output_ids = output_ids[:, :max_len]
                    print(f"Adjusted input and label lengths to {max_len}")
                    input_ids = input_ids.to(device)
                    output_ids = output_ids.to(device)
                
                inputs['input_ids'] = input_ids
                inputs['attention_mask'] = input_ids.ne(0).to(device)


                # Calculate loss for current batch and perform gradient accumulation
                with torch.enable_grad():
                    outputs = model(**inputs, labels=output_ids)
                    loss = outputs[0]
                    loss = loss / accumulation_steps  # Normalize loss
                    loss.backward()
            else: # LLaVA v1.5
                # Use message_to_prompt_train to process batch data
                input_ids, images, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                
                # Define recursive function to move nested tensor structures to specified device
                def move_to_device(obj, target_device):
                    if isinstance(obj, torch.Tensor):
                        return obj.to(target_device)
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(move_to_device(item, target_device) for item in obj)
                    elif isinstance(obj, dict):
                        return {k: move_to_device(v, target_device) for k, v in obj.items()}
                    else:
                        return obj
                
                # Move data to corresponding device
                input_ids = move_to_device(input_ids, device)
                image_sizes = None
                if images is not None:
                    images, image_sizes = images
                    images = move_to_device(images, device)
                output_ids = move_to_device(output_ids, device)
                
                # Adjust input and label lengths to match
                if input_ids.size(1) != output_ids.size(1):
                    max_len = max(input_ids.size(1), output_ids.size(1))
                    if input_ids.size(1) < max_len:
                        padding = torch.zeros((input_ids.size(0), max_len - input_ids.size(1)), 
                                            dtype=input_ids.dtype, device=input_ids.device)
                        input_ids = torch.cat([input_ids, padding], dim=1)
                    else:
                        input_ids = input_ids[:, :max_len]
                    if output_ids.size(1) < max_len:
                        padding = torch.full((output_ids.size(0), max_len - output_ids.size(1)), 
                                            fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
                        output_ids = torch.cat([output_ids, padding], dim=1)
                    else:
                        output_ids = output_ids[:, :max_len]
                    logging.info(f"Adjusted input and label lengths to {max_len}")
                    input_ids = input_ids.to(device)
                    output_ids = output_ids.to(device)
                else:
                    logging.info(f"Original input and label lengths: {input_ids.size(1)}")
                
                attention_mask = input_ids.ne(0).to(device) 
                # breakpoint()
                # Calculate loss for current batch and perform gradient accumulation
                with torch.enable_grad():
                    outputs = model(input_ids=input_ids, images=images, labels=output_ids, attention_mask=attention_mask, image_sizes=image_sizes)
                    loss = outputs[0]
                    loss = loss / accumulation_steps  # Normalize loss
                    loss.backward()
            
            batch_count += 1  # Count each batch
            
            # Perform backpropagation when accumulated the set number of batches
            if batch_count % accumulation_steps == 0:
                model.zero_grad()  # Clear gradients
        
        except Exception as e:
            print(f"Error occurred during Grad information calculation: {e}")
            import traceback
            print("Detailed error information:")
            traceback.print_exc()
            if isinstance(batch, dict):
                print(f"Batch data keys: {list(batch.keys())}")
            elif isinstance(batch, list) and len(batch) > 0:
                print(f"Type of first item in batch data: {type(batch[0])}")
            continue
        
    # Remove registered hooks
    for h in list(handles.values()):
        h.remove()
    for h in list(empirical_forward_handles.values()):
        h.remove()
    logging.info('finished output gradient distribution computing')
    # Debug: report whether output_grad_distribution hooks were triggered
    triggered = sum(1 for c in hook_trigger_count.values() if c > 0)
    total_hooks = len(handles)
    logging.info(f"output_grad_distribution hooks: triggered={triggered}/{total_hooks}, batch_count={batch_count}")
    if triggered < total_hooks and total_hooks > 0:
        untriggered = [k for k in handles if hook_trigger_count.get(k, 0) == 0]
        logging.warning(f"output_grad_distribution hooks never triggered for: {untriggered[:10]}{'...' if len(untriggered) > 10 else ''}")
    # Cholesky Decomposition (skip modules where hook never ran, e.g. output_grad_distribution still 0)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _is_lm_layer_name(name):
            if hasattr(module, 'output_grad_distribution'):
                if isinstance(module.output_grad_distribution, torch.Tensor):
                    G = module.output_grad_distribution.double().to(utils.get_dev())
                    diag_shift = torch.zeros(
                        G.shape[:-2] if G.ndim > 2 else (),
                        device=G.device,
                        dtype=G.dtype,
                    )
                    try:
                        output_grad_scaling_matrix = torch.linalg.cholesky(G, upper=True).float()
                    except Exception as e:
                        print("Warning: eigen output_grad_distribution is not positive!")
                        G = (G + G.transpose(-1, -2)) / 2
                        eigenvalues = torch.linalg.eigvalsh(G) # [..., D]
                        min_eig = eigenvalues[..., 0] # 每个批里的最小特征值
                        eps = 1e-6
                        shift = (-min_eig + eps).clamp_min(0.0) # 只对非正定的批次加抖动
                        diag_shift = shift
                        I = torch.eye(G.size(-1), device=G.device, dtype=G.dtype)  # [D, D]，可广播到批维
                        G = G + shift[..., None, None] * I               # [..., D, D]
                        output_grad_scaling_matrix = torch.linalg.cholesky(G, upper=True).float()
                        del eigenvalues, min_eig, eps, shift, I
                    if grad_debug_payload is not None:
                        grad_debug_payload[name] = debug_utils.grad_calib_debug_pack(
                            raw_cpu_double=G.detach().cpu(),
                            chol_cpu=output_grad_scaling_matrix.detach().cpu(),
                            normalization_count=max(getattr(module, "_debug_token_count", 0), 1),
                            sample_count=getattr(module, "_debug_sample_count", 0),
                            token_count=getattr(module, "_debug_token_count", 0),
                            hook_count=getattr(module, "_debug_hook_count", hook_trigger_count.get(name, 0)),
                            diag_shift_cpu=diag_shift.detach().cpu(),
                        )
                        empirical_cfg = getattr(module, "_empirical_fisher_cfg", None)
                        if empirical_cfg is not None:
                            grad_debug_payload[name]["empirical_fisher_debug"] = debug_utils.empirical_fisher_debug_pack(empirical_cfg)
                    module.output_grad_scaling_matrix = output_grad_scaling_matrix.cpu()
                    module.output_grad_distribution = None
                    del G, output_grad_scaling_matrix
                    torch.cuda.empty_cache()
                else:
                    # Hook never accumulated (e.g. all batches failed); clear placeholder
                    module.output_grad_distribution = None
                if hasattr(module, "_debug_sample_count"):
                    del module._debug_sample_count
                if hasattr(module, "_debug_token_count"):
                    del module._debug_token_count
                if hasattr(module, "_debug_hook_count"):
                    del module._debug_hook_count
                if hasattr(module, "_empirical_fisher_input_proj"):
                    del module._empirical_fisher_input_proj
                if hasattr(module, "_empirical_fisher_cfg"):
                    del module._empirical_fisher_cfg
    logging.info('finished output gradient distribution Cholesky decomposition')
    # Save output gradient scaling matrix
    all_output_grad_scaling_matrix = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _is_lm_layer_name(name):
            if hasattr(module, 'output_grad_scaling_matrix'):
                all_output_grad_scaling_matrix[name] = module.output_grad_scaling_matrix.cpu()

    logging.info(f"Saving output gradient scaling matrix cache to {cache_file}...")
    torch.save(all_output_grad_scaling_matrix, cache_file)
    logging.info("Output gradient scaling matrix cache saved successfully!")
    if grad_debug_payload is not None:
        debug_utils.save_act_calib_debug_payload(
            args,
            filename="act_calib_debug_grad",
            payload=grad_debug_payload,
            log_prefix="act_calib_debug",
        )
    

@torch.enable_grad()
def calib_grad_distribution_low_resources(model, dataloader, tokenizer, image_processor, args, use_cache=True, cache_file=None):
    """
    Calculate Grad distribution for each layer's output
    
    Args:
        model: Model to be calibrated
        tokenizer: Tokenizer
        image_processor: Image processor
        args: Parameter configuration
        use_cache: Whether to use cache
        cache_file: Cache file path, automatically generated if None
    """
    logging.info("Starting output gradient distribution calculation for low resources...")
    model_id = model.config._name_or_path
    
    if cache_file is None:
        cache_dir  = args.act_cache_dir + "/cache"
        if args.cache_in_log:
            cache_dir = args.act_cache_dir + "/cache"
        os.makedirs(cache_dir, exist_ok=True)
        # Add relevant information to cache file name
        if args.qkv_fuse:
            raise ValueError("QKV fuse is not supported for output gradient distribution")
        elif args.kv_fuse:
            raise ValueError("KV fuse is not supported for output gradient distribution")
        else:
            fusion_mode = "separate"
        if bool(getattr(args, "pre_svd_rotate", False)):
            rotate_info = "presvdrot"
        else:
            rotate_info = "presvdnorot"
            
        method = args.calib_method if hasattr(args, "act_aware") and args.act_aware else "no_act_aware"
        if method != "cholesky":
            raise ValueError("Only cholesky calibration method is supported for output gradient distribution")
        
        if args.misalign_text_image:
            fusion_mode = fusion_mode + "_misalign"
        if args.is_per_head_svd:
            fusion_mode = fusion_mode + "_perhead"
        if args.taylor_order != 2:
            fusion_mode = fusion_mode + "_taylor" + str(args.taylor_order)
        calib_tag = _act_calib_cache_tag(args) + reasoning_calib_utils.cache_tag(
            args,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
        )
        if args.a_clip_ratio == 1.0:
            cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_{args.nsamples}_{args.seed}_{rotate_info}_{args.svd_modules}_{fusion_mode}{calib_tag}_calib_output_grad_distribution_{method}_args{args.svd_modules}.pt")
        else:
            cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_aclip{args.a_clip_ratio}_{args.nsamples}_{args.seed}_{rotate_info}_{args.svd_modules}_{fusion_mode}{calib_tag}_calib_output_grad_distribution_{method}_args{args.svd_modules}.pt")
    
    cache_file = calib_cache.released_file(args, "output_grad") or cache_file
    if os.path.exists(cache_file) and use_cache:
        logging.info(f"Loading output gradient distribution cache from {cache_file}...")
        all_output_grad_scaling_matrix = torch.load(cache_file, map_location="cpu")
        # Load output gradient distribution into the self_attn.S_grad_info attribute of corresponding layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'lm_head' not in name and _is_lm_layer_name(name):
                if name in all_output_grad_scaling_matrix:
                    if args.qkv_fuse:
                        raise ValueError("QKV fuse is not supported for output gradient distribution")
                    elif args.kv_fuse:
                        raise ValueError("KV fuse is not supported for output gradient distribution")
                    else:
                        # for separate SVD, output gradient distribution is on q_proj, k_proj, v_proj
                        module.output_grad_scaling_matrix = all_output_grad_scaling_matrix[name].to(utils.get_dev())
        logging.info("Successfully loaded output gradient distribution cache!")
        if getattr(args, "act_calib_debug", False):
            logging.info(
                "act_calib_debug: cache hit for output-gradient calibration — raw covariance tensors were not recomputed; "
                "run with --no-use_cache or use_cache=False for online grad debug dump."
            )
        return

    print("Starting output gradient distribution calculation...")
    logging.info('start output gradient distribution computing')
    model.eval()

    # Ensure the entire model is on CUDA
    device = utils.get_dev()
    model = model.to(device)
    reasoning_batches = None
    if reasoning_calib_utils.enabled(args):
        reasoning_batches = reasoning_calib_utils.materialize_reasoning_batches(
            model=model,
            dataloader=dataloader,
            tokenizer=tokenizer,
            image_processor=image_processor,
            args=args,
            device=device,
        )

    # Set model to training mode and only allow gradient computation for svd_modules layers
    model.train()
    
    # only collect output gradient, not weight gradient
    # thus we can set weight gradient only for layer 0's q_proj, k_proj, v_proj
    # 先关掉所有参数梯度
    for _, p in model.named_parameters():
        p.requires_grad = False

    # 按索引开启需要梯度的少量层（比如只开第 0 层）
    layers = model_utils.get_layers(model)
    num_layers = len(layers)
    attn = layers[0].self_attn
    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
        if hasattr(attn, proj_name):
            for p in getattr(attn, proj_name).parameters():
                p.requires_grad = True

    handles = {}
    module_name_lookup = {id(module): name for name, module in model.named_modules()}
    num_heads = layers[0].self_attn.config.num_attention_heads #model.config.num_attention_heads
    num_heads_kv = layers[0].self_attn.config.num_key_value_heads
    head_dim = layers[0].self_attn.config.hidden_size // num_heads
    
    all_output_grad_scaling_matrix = {}
    grad_debug_payload = {} if getattr(args, "act_calib_debug", False) else None
    empirical_fisher_active = _empirical_fisher_enabled(args)
    empirical_target_layers, empirical_layer_strategy = _empirical_fisher_target_layers(args, num_layers)
    empirical_target_layers = set(empirical_target_layers)
    empirical_forward_handles = {}
    
    def make_output_grad_hook(name):
        def hook(module, grad_in, grad_out):
            bsz = grad_out[0].shape[0]
            base_grad = grad_out[0].detach()
            token_count = int(base_grad.numel() // max(int(base_grad.shape[-1]), 1))
            sample_count = int(base_grad.shape[0] if base_grad.dim() > 1 else 1)
            module._debug_hook_count = int(getattr(module, "_debug_hook_count", 0) + 1)
            module._debug_sample_count = int(getattr(module, "_debug_sample_count", 0) + sample_count)
            module._debug_token_count = int(getattr(module, "_debug_token_count", 0) + token_count)
            if args.is_per_head_svd and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name):
                # [batch_size, num_heads, seq_len, head_dim]
                if 'q_proj' in name:
                    grads = grad_out[0].detach().view(bsz, -1, num_heads, head_dim).permute(0, 2, 1, 3).contiguous().float()
                elif 'k_proj' in name:
                    grads = grad_out[0].detach().view(bsz, -1, num_heads_kv, head_dim).permute(0, 2, 1, 3).contiguous().float()
                elif 'v_proj' in name:
                    grads = grad_out[0].detach().view(bsz, -1, num_heads_kv, head_dim).permute(0, 2, 1, 3).contiguous().float()
                else:
                    raise ValueError(f"Invalid projection name: {name}")
            else:
                # [batch_size, seq_len, hidden_size]
                grads = grad_out[0].detach().float()
            # calculate covariance of output gradients
            grads_cov = torch.matmul(grads.transpose(-1, -2), grads).sum(dim=0) # [hidden_size, hidden_size] or [num_heads, head_dim, head_dim]
            if not hasattr(module, "output_grad_distribution"):
                module.register_buffer("output_grad_distribution", torch.zeros_like(grads_cov, dtype=torch.float32))
            module.output_grad_distribution += grads_cov
            empirical_cfg = getattr(module, "_empirical_fisher_cfg", None)
            input_proj = getattr(module, "_empirical_fisher_input_proj", None)
            if empirical_cfg is not None and isinstance(input_proj, torch.Tensor) and input_proj.numel() > 0:
                for head_idx, head_state in empirical_cfg["heads"].items():
                    if grads.ndim < 4 or int(head_idx) >= int(grads.shape[1]):
                        continue
                    output_projection = head_state["output_projection"].to(device=grads.device, dtype=grads.dtype)
                    grad_head = grads[:, int(head_idx), :, :]
                    grad_proj = torch.einsum("bth,hk->btk", grad_head, output_projection)
                    sketch = torch.einsum("btk,btr->bkr", grad_proj, input_proj)
                    flat = sketch.reshape(sketch.shape[0], -1).detach().float().cpu()
                    head_state["sketch_outer"] += flat.transpose(0, 1) @ flat
                    head_state["sketch_diag"] += torch.sum(flat * flat, dim=0)
                    head_state["sketch_trace_sum"] += float(torch.sum(flat * flat).item())
                    head_state["sample_count"] += int(flat.shape[0])
                    empirical_cfg["sample_count"] += int(flat.shape[0])
                delattr(module, "_empirical_fisher_input_proj")
            del grads, grads_cov
        return hook

    def make_empirical_input_hook(module_name):
        def hook(module, inputs, output):
            empirical_cfg = getattr(module, "_empirical_fisher_cfg", None)
            if empirical_cfg is None or not inputs:
                return
            inp = inputs[0].detach()
            if not isinstance(inp, torch.Tensor):
                return
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            inp = inp.float()
            input_projection = empirical_cfg["input_projection"].to(device=inp.device, dtype=inp.dtype)
            module._empirical_fisher_input_proj = torch.einsum("btd,dk->btk", inp, input_projection).detach()
        return hook
    
    for idx in tqdm(range(num_layers), desc="Computing Output Gradient Distributions Low Resources"):
        # Register backward hooks for selected layer indices
        # args.grad_hook_layer_idx can be int, list/tuple of int, or comma-separated string, e.g., "0,3,5"
        # args.grad_hook_layer_idx = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
        # args.grad_hook_layer_idx = [idx]
        # if hasattr(args, 'grad_hook_layer_idx'):
        #     raw_idx = args.grad_hook_layer_idx
        #     if isinstance(raw_idx, (list, tuple)):
        #         target_indices = [int(i) for i in raw_idx]
        #     else:
        #         raw_str = str(raw_idx).strip()
        #         target_indices = [int(x) for x in raw_str.split(',') if x != '']
        # else:
        #     target_indices = []
        target_indices = [idx]

        current_modules = []
        if len(target_indices) > 0:
            for idx, layer in enumerate(model_utils.get_layers(model)):
                if idx not in target_indices:
                    continue
                if args.svd_modules in ['qkv', 'attn', 'all']:
                    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                        if hasattr(layer.self_attn, proj_name):
                            proj = getattr(layer.self_attn, proj_name)
                            key = f"layer_{idx}.{proj_name}"
                            handles[key] = proj.register_full_backward_hook(make_output_grad_hook(key))
                            proj.output_grad_distribution = 0
                            module_name = module_name_lookup.get(id(proj), key)
                            current_modules.append((module_name, proj))
                            if empirical_fisher_active and idx in empirical_target_layers:
                                num_proj_heads = num_heads if proj_name == 'q_proj' else num_heads_kv
                                selected_heads, head_strategy = _empirical_fisher_target_heads(args, num_proj_heads)
                                if selected_heads:
                                    proj._empirical_fisher_cfg = _empirical_fisher_module_config(
                                        args=args,
                                        module_name=module_name,
                                        layer_idx=idx,
                                        proj_name=proj_name,
                                        in_features=proj.in_features,
                                        head_dim=head_dim,
                                        selected_heads=selected_heads,
                                        device=proj.weight.device,
                                    )
                                    proj._empirical_fisher_cfg["selection_strategy"] = f"{empirical_layer_strategy}_{head_strategy}"
                                    empirical_forward_handles[key] = proj.register_forward_hook(make_empirical_input_hook(module_name))
                if args.svd_modules in ['o_proj', 'attn', 'all']:
                    if hasattr(layer.self_attn, 'o_proj'):
                        proj = layer.self_attn.o_proj
                        key = f"layer_{idx}.o_proj"
                        handles[key] = proj.register_full_backward_hook(make_output_grad_hook(key))
                        proj.output_grad_distribution = 0
                        module_name = module_name_lookup.get(id(proj), key)
                        current_modules.append((module_name, proj))
                if args.svd_modules in ['gaup', 'mlp', 'all']:
                    for proj_name in ['up_proj', 'gate_proj']:
                        if hasattr(layer.mlp, proj_name):
                            proj = getattr(layer.mlp, proj_name)
                            key = f"layer_{idx}.{proj_name}"
                            handles[key] = proj.register_full_backward_hook(make_output_grad_hook(key))
                            proj.output_grad_distribution = 0
                            module_name = module_name_lookup.get(id(proj), key)
                            current_modules.append((module_name, proj))
                if args.svd_modules in ['down', 'mlp', 'all']:
                    if hasattr(layer.mlp, 'down_proj'):
                        proj = layer.mlp.down_proj
                        key = f"layer_{idx}.down_proj"
                        handles[key] = proj.register_full_backward_hook(make_output_grad_hook(key))
                        proj.output_grad_distribution = 0
                        module_name = module_name_lookup.get(id(proj), key)
                        current_modules.append((module_name, proj))
        
        accumulation_steps = 1   # Number of accumulated batches
        batch_count = 0          # Accumulated batch counter
        
        grad_iter = reasoning_batches if reasoning_batches is not None else dataloader
        for batch_idx, batch in enumerate(tqdm(grad_iter, desc="Computing Output Gradient Distributions")):
            try:
                is_qwen_model = grad_info_utils._is_qwen_model(args, model, tokenizer)
                if reasoning_batches is not None:
                    reasoning_batch = batch
                    with torch.enable_grad():
                        outputs = reasoning_calib_utils.forward_reasoning_batch(model, reasoning_batch)
                        loss = reasoning_calib_utils.compute_reasoning_loss(outputs, reasoning_batch, args)
                        loss = loss / accumulation_steps
                        loss.backward()
                elif tokenizer is None and image_processor is None:
                    input_ids = batch[0].to(device)
                    labels = input_ids.clone()
                    if args.label_shift: # enable label shift like palu, or spinquant handle this in Trainer?
                        labels = labels[:, 1:]
                        input_ids = input_ids[:, :-1]
                    with torch.enable_grad():
                        outputs = model(input_ids, labels=labels)
                        loss = outputs[0]
                        loss = loss / accumulation_steps  # Normalize loss
                        loss.backward()
                elif is_qwen_model:
                    inputs, labels = grad_info_utils._build_qwen_train_inputs(batch, image_processor, device, args=args)
                    with torch.enable_grad():
                        outputs = model(**inputs, labels=labels)
                        loss = outputs[0]
                        loss = loss / accumulation_steps
                        loss.backward()
                elif molmo_utils.is_molmo_model(args=args, model=model, tokenizer=tokenizer):
                    inputs, labels = molmo_utils.build_molmo_train_inputs(
                        batch,
                        image_processor,
                        device,
                        model=model,
                        label_mode=args.label_mode,
                        label_shift=bool(getattr(args, "label_shift", True)),
                    )
                    with torch.enable_grad():
                        outputs = model(**inputs, labels=labels)
                        loss = outputs[0]
                        loss = loss / accumulation_steps
                        loss.backward()
                elif tokenizer is None: # SmolVLM
                        # Use message_to_prompt_train to process batch data
                    inputs, _, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                    # Define recursive function to move nested tensor structures to specified device
                    def move_to_device(obj, target_device):
                        if isinstance(obj, torch.Tensor):
                            return obj.to(target_device)
                        elif isinstance(obj, (list, tuple)):
                            return type(obj)(move_to_device(item, target_device) for item in obj)
                        elif isinstance(obj, dict):
                            return {k: move_to_device(v, target_device) for k, v in obj.items()}
                        else:
                            return obj
                    
                    # Move data to corresponding device
                    inputs = move_to_device(inputs, device)
                    output_ids = move_to_device(output_ids, device)

                    input_ids = inputs.get('input_ids')
                    if not isinstance(output_ids, torch.Tensor) and hasattr(output_ids, 'get'):
                        output_ids = output_ids.get('input_ids', output_ids)
                    if isinstance(output_ids, dict):
                        output_ids = output_ids.get('input_ids')
                    
                    # Adjust input and label lengths to match
                    # breakpoint()
                    if input_ids.size(1) != output_ids.size(1):
                        max_len = max(input_ids.size(1), output_ids.size(1))
                        if input_ids.size(1) < max_len:
                            padding = torch.zeros((input_ids.size(0), max_len - input_ids.size(1)), 
                                                dtype=input_ids.dtype, device=input_ids.device)
                            input_ids = torch.cat([input_ids, padding], dim=1)
                        else:
                            input_ids = input_ids[:, :max_len]
                        if output_ids.size(1) < max_len:
                            padding = torch.full((output_ids.size(0), max_len - output_ids.size(1)), 
                                                fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
                            output_ids = torch.cat([output_ids, padding], dim=1)
                        else:
                            output_ids = output_ids[:, :max_len]
                        print(f"Adjusted input and label lengths to {max_len}")
                        input_ids = input_ids.to(device)
                        output_ids = output_ids.to(device)
                    
                    inputs['input_ids'] = input_ids
                    inputs['attention_mask'] = input_ids.ne(0).to(device)


                    # Calculate loss for current batch and perform gradient accumulation
                    with torch.enable_grad():
                        outputs = model(**inputs, labels=output_ids)
                        loss = outputs[0]
                        loss = loss / accumulation_steps  # Normalize loss
                        loss.backward()
                elif tokenizer == 'hf_v16': # LLaVA Next/v1.6
                        # Use message_to_prompt_train to process batch data
                    inputs, _, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                    
                    # Define recursive function to move nested tensor structures to specified device
                    def move_to_device(obj, target_device):
                        if isinstance(obj, torch.Tensor):
                            return obj.to(target_device)
                        elif isinstance(obj, (list, tuple)):
                            return type(obj)(move_to_device(item, target_device) for item in obj)
                        elif isinstance(obj, dict):
                            return {k: move_to_device(v, target_device) for k, v in obj.items()}
                        else:
                            return obj
                    
                    # Move data to corresponding device
                    inputs = move_to_device(inputs, device)
                    output_ids = move_to_device(output_ids, device)
                    input_ids = inputs.get('input_ids')
                    if not isinstance(output_ids, torch.Tensor) and hasattr(output_ids, 'get'):
                        output_ids = output_ids.get('input_ids', output_ids)
                    if isinstance(output_ids, dict):
                        output_ids = output_ids.get('input_ids')

                    # breakpoint()
                    # 0 as pad token id, following many tokenizers
                    if input_ids.size(1) != output_ids.size(1):
                        max_len = max(input_ids.size(1), output_ids.size(1))
                        if input_ids.size(1) < max_len:
                            padding = torch.zeros((input_ids.size(0), max_len - input_ids.size(1)), 
                                                dtype=input_ids.dtype, device=input_ids.device)
                            input_ids = torch.cat([input_ids, padding], dim=1)
                        else:
                            input_ids = input_ids[:, :max_len]
                        if output_ids.size(1) < max_len:
                            padding = torch.full((output_ids.size(0), max_len - output_ids.size(1)), 
                                                fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
                            output_ids = torch.cat([output_ids, padding], dim=1)
                        else:
                            output_ids = output_ids[:, :max_len]
                        print(f"Adjusted input and label lengths to {max_len}")
                        input_ids = input_ids.to(device)
                        output_ids = output_ids.to(device)
                    
                    inputs['input_ids'] = input_ids
                    inputs['attention_mask'] = input_ids.ne(0).to(device)


                    # Calculate loss for current batch and perform gradient accumulation
                    with torch.enable_grad():
                        outputs = model(**inputs, labels=output_ids)
                        loss = outputs[0]
                        loss = loss / accumulation_steps  # Normalize loss
                        loss.backward()
                else: # LLaVA v1.5
                    # Use message_to_prompt_train to process batch data
                    input_ids, images, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                    
                    # Define recursive function to move nested tensor structures to specified device
                    def move_to_device(obj, target_device):
                        if isinstance(obj, torch.Tensor):
                            return obj.to(target_device)
                        elif isinstance(obj, (list, tuple)):
                            return type(obj)(move_to_device(item, target_device) for item in obj)
                        elif isinstance(obj, dict):
                            return {k: move_to_device(v, target_device) for k, v in obj.items()}
                        else:
                            return obj
                    
                    # Move data to corresponding device
                    input_ids = move_to_device(input_ids, device)
                    image_sizes = None
                    if images is not None:
                        images, image_sizes = images
                        images = move_to_device(images, device)
                    output_ids = move_to_device(output_ids, device)
                    
                    # Adjust input and label lengths to match
                    if input_ids.size(1) != output_ids.size(1):
                        max_len = max(input_ids.size(1), output_ids.size(1))
                        if input_ids.size(1) < max_len:
                            padding = torch.zeros((input_ids.size(0), max_len - input_ids.size(1)), 
                                                dtype=input_ids.dtype, device=input_ids.device)
                            input_ids = torch.cat([input_ids, padding], dim=1)
                        else:
                            input_ids = input_ids[:, :max_len]
                        if output_ids.size(1) < max_len:
                            padding = torch.full((output_ids.size(0), max_len - output_ids.size(1)), 
                                                fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
                            output_ids = torch.cat([output_ids, padding], dim=1)
                        else:
                            output_ids = output_ids[:, :max_len]
                        logging.info(f"Adjusted input and label lengths to {max_len}")
                        input_ids = input_ids.to(device)
                        output_ids = output_ids.to(device)
                    else:
                        logging.info(f"Original input and label lengths: {input_ids.size(1)}")
                    
                    attention_mask = input_ids.ne(0).to(device) 
                    # breakpoint()
                    # Calculate loss for current batch and perform gradient accumulation
                    with torch.enable_grad():
                        outputs = model(input_ids=input_ids, images=images, labels=output_ids, attention_mask=attention_mask, image_sizes=image_sizes)
                        loss = outputs[0]
                        loss = loss / accumulation_steps  # Normalize loss
                        loss.backward()
                
                batch_count += 1  # Count each batch
                
                # Perform backpropagation when accumulated the set number of batches
                if batch_count % accumulation_steps == 0:
                    model.zero_grad()  # Clear gradients
            
            except Exception as e:
                print(f"Error occurred during Grad information calculation: {e}")
                import traceback
                print("Detailed error information:")
                traceback.print_exc()
                if isinstance(batch, dict):
                    print(f"Batch data keys: {list(batch.keys())}")
                elif isinstance(batch, list) and len(batch) > 0:
                    print(f"Type of first item in batch data: {type(batch[0])}")
                continue
            
        # Remove registered hooks
        for h in list(handles.values()):
            h.remove()
        handles.clear()
        for h in list(empirical_forward_handles.values()):
            h.remove()
        empirical_forward_handles.clear()
        logging.info('finished output gradient distribution computing')
        # Cholesky Decomposition (skip modules where hook never ran)
        for name, module in current_modules:
            if isinstance(module, nn.Linear) and _is_lm_layer_name(name):
                if hasattr(module, 'output_grad_distribution'):
                    if isinstance(module.output_grad_distribution, torch.Tensor):
                        G = module.output_grad_distribution.double().to(utils.get_dev())
                        diag_shift = torch.zeros(
                            G.shape[:-2] if G.ndim > 2 else (),
                            device=G.device,
                            dtype=G.dtype,
                        )
                        try:
                            output_grad_scaling_matrix = torch.linalg.cholesky(G, upper=True).float()
                        except Exception as e:
                            print("Warning: eigen output_grad_distribution is not positive!")
                            G = (G + G.transpose(-1, -2)) / 2
                            eigenvalues = torch.linalg.eigvalsh(G) # [..., D]
                            min_eig = eigenvalues[..., 0] # 每个批里的最小特征值
                            eps = 1e-6
                            shift = (-min_eig + eps).clamp_min(0.0) # 只对非正定的批次加抖动
                            diag_shift = shift
                            I = torch.eye(G.size(-1), device=G.device, dtype=G.dtype)  # [D, D]，可广播到批维
                            G = G + shift[..., None, None] * I               # [..., D, D]
                            output_grad_scaling_matrix = torch.linalg.cholesky(G, upper=True).float()
                            del eigenvalues, min_eig, eps, shift, I
                        if grad_debug_payload is not None:
                            grad_debug_payload[name] = debug_utils.grad_calib_debug_pack(
                                raw_cpu_double=G.detach().cpu(),
                                chol_cpu=output_grad_scaling_matrix.detach().cpu(),
                                normalization_count=max(getattr(module, "_debug_token_count", 0), 1),
                                sample_count=getattr(module, "_debug_sample_count", 0),
                                token_count=getattr(module, "_debug_token_count", 0),
                                hook_count=getattr(module, "_debug_hook_count", 0),
                                diag_shift_cpu=diag_shift.detach().cpu(),
                            )
                            empirical_cfg = getattr(module, "_empirical_fisher_cfg", None)
                            if empirical_cfg is not None:
                                grad_debug_payload[name]["empirical_fisher_debug"] = debug_utils.empirical_fisher_debug_pack(empirical_cfg)
                        module.output_grad_scaling_matrix = output_grad_scaling_matrix.cpu()
                        module.output_grad_distribution = None
                        del G, output_grad_scaling_matrix
                        torch.cuda.empty_cache()
                    else:
                        module.output_grad_distribution = None
                    if hasattr(module, "_debug_sample_count"):
                        del module._debug_sample_count
                    if hasattr(module, "_debug_token_count"):
                        del module._debug_token_count
                    if hasattr(module, "_debug_hook_count"):
                        del module._debug_hook_count
                    if hasattr(module, "_empirical_fisher_input_proj"):
                        del module._empirical_fisher_input_proj
                    if hasattr(module, "_empirical_fisher_cfg"):
                        del module._empirical_fisher_cfg
        logging.info('finished output gradient distribution Cholesky decomposition')
        # Save output gradient scaling matrix
        
        for name, module in current_modules:
            if isinstance(module, nn.Linear) and _is_lm_layer_name(name):
                if hasattr(module, 'output_grad_scaling_matrix'):
                    all_output_grad_scaling_matrix[name] = module.output_grad_scaling_matrix.cpu()
        torch.cuda.empty_cache()

    logging.info(f"Saving output gradient scaling matrix cache to {cache_file}...")
    torch.save(all_output_grad_scaling_matrix, cache_file)
    logging.info("Output gradient scaling matrix cache saved successfully!")
    if grad_debug_payload is not None:
        debug_utils.save_act_calib_debug_payload(
            args,
            filename="act_calib_debug_grad",
            payload=grad_debug_payload,
            log_prefix="act_calib_debug (low-resource)",
        )
    
def svd_with_loss_aware(layers, args, use_cache=True, cache_file=None):
    """
    Perform loss-aware singular value truncation on linear layers
    
    Args:
        layers: List of model layers
        args: Parameter configuration
        use_cache: Whether to use cache
        cache_file: Cache file path, automatically generated if None
    
    Returns:
        singular_values_dict: Dictionary containing singular values for each layer
    """
    if bool(getattr(args, "truncate_hybrid_grad_layer_budget", False)) and bool(
        getattr(args, "truncate_hybrid_loss_aware_layer_budget", False)
    ):
        raise ValueError(
            "--truncate_hybrid_grad_layer_budget and "
            "--truncate_hybrid_loss_aware_layer_budget cannot both be enabled."
        )

    calib_method_info = args.calib_method if hasattr(args, "act_aware") and args.act_aware else "no_act_aware"
    if args.qkv_fuse:
        fusion_mode = "qkv_fuse"
    elif args.kv_fuse:
        fusion_mode = "kv_fuse"
    else:
        fusion_mode = "separate"
    if bool(getattr(args, "pre_svd_rotate", False)):
        rotate_info = "presvdrot"
    else:
        rotate_info = "presvdnorot"
    if hasattr(args, "use_S_gradinfo_init") and args.use_S_gradinfo_init:
        fusion_mode += "_S_gradinfo_init"
    if args.misalign_text_image:
        fusion_mode = fusion_mode + "_misalign"
    if args.is_per_head_svd:
        fusion_mode = fusion_mode + "_perhead"
    fusion_mode = fusion_mode + reasoning_calib_utils.cache_tag(args)

    def _build_sv_cache_path(cache_dir):
        a_bits_calib = args.a_bits_calib if hasattr(args, "a_bits_calib") else 16
        base_name = f"{args.model.replace('/','_')}_{args.nsamples}_{args.seed}_{calib_method_info}_{rotate_info}_{args.svd_modules}_{fusion_mode}_acalib{a_bits_calib}_singular_values.pt"
        if hasattr(args, "a_clip_ratio") and args.a_clip_ratio != 1.0:
            base_name = f"{args.model.replace('/','_')}_aclip{args.a_clip_ratio}_{args.nsamples}_{args.seed}_{calib_method_info}_{rotate_info}_{args.svd_modules}_{fusion_mode}_acalib{a_bits_calib}_singular_values.pt"
        return os.path.join(cache_dir, base_name)

    # Automatically generate cache file path
    if cache_file is None:
        cache_dir  = args.model + "/cache"
        if hasattr(args, "cache_in_log") and args.cache_in_log:
            cache_dir = args.save_path + "/cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = _build_sv_cache_path(cache_dir)
    
    # If cache exists and cache is enabled, load directly
    if os.path.exists(cache_file) and use_cache:
        logging.info(f"Loading singular values cache from {cache_file}...")
        singular_values_dict = torch.load(cache_file, map_location="cpu")
        logging.info("Successfully loaded singular values cache!")
        for idx, layer in enumerate(layers):
                if args.svd_modules in ['all', 'mlp', 'gaup']:
                    if args.mlp_fuse:
                        if hasattr(layer.mlp.up_proj, "svd_info"):
                            del layer.mlp.up_proj.svd_info
                    else:
                        if hasattr(layer.mlp.up_proj, "svd_info"):
                            del layer.mlp.up_proj.svd_info
                        if hasattr(layer.mlp.gate_proj, "svd_info"):
                            del layer.mlp.gate_proj.svd_info
                if args.svd_modules in ['all', 'mlp', 'down']:
                    if hasattr(layer.mlp.down_proj, "svd_info"):
                        del layer.mlp.down_proj.svd_info
                logging.info(f"Successfully cleaned svd_info for layer {idx}!, quick fix for grad score cache existing")

    else:
        # Load gradient information cache file
        singular_values_cache_dir  = args.model + "/cache" # still have differenciate issue with VIT different quant configuration
        if hasattr(args, "cache_in_log") and args.cache_in_log:
            singular_values_cache_dir = args.save_path + "/cache"
        # Build gradient information cache file path
        singular_values_cache = _build_sv_cache_path(singular_values_cache_dir)
        
        # Check if gradient information cache exists
        if os.path.exists(singular_values_cache):
            logging.info(f"Loading singular values from {singular_values_cache}...")
            singular_values_dict = torch.load(singular_values_cache, map_location="cpu")
            
            # Load singular values into corresponding layers
            for idx, layer in enumerate(layers):
                layer_key = f"layer_{idx}"
                if layer_key in singular_values_dict:
                    if not hasattr(layer.self_attn.k_proj, 'singular_values'):
                        if args.svd_modules in ['qkv', 'attn', 'all', 'all_sep_fine', 'all_sep_fine_wo_down', 'all_sep']:
                            if args.qkv_fuse:
                                layer.self_attn.k_proj.singular_values = singular_values_dict[layer_key]['k_proj'].to(utils.get_dev())
                            elif args.kv_fuse:
                                # Load Q separately and KV together
                                if 'q_proj' in singular_values_dict[layer_key]:
                                    layer.self_attn.q_proj.singular_values = singular_values_dict[layer_key]['q_proj'].to(utils.get_dev())
                                if 'k_proj' in singular_values_dict[layer_key]:
                                    layer.self_attn.k_proj.singular_values = singular_values_dict[layer_key]['k_proj'].to(utils.get_dev())
                            # elif args.inp_fuse:
                            #     layer.self_attn.k_proj.singular_values = singular_values_dict[layer_key]['k_proj'].to(utils.get_dev())
                            else:
                                # Separate Q, K, V
                                for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                                    if proj_name in singular_values_dict[layer_key]:
                                        proj = getattr(layer.self_attn, proj_name)
                                        proj.singular_values = singular_values_dict[layer_key][proj_name].to(utils.get_dev())
                        if args.svd_modules in ['attn', 'all', 'o', 'all_sep_fine', 'all_sep_fine_wo_down', 'all_sep']:
                            layer.self_attn.o_proj.singular_values = singular_values_dict[layer_key]['o_proj'].to(utils.get_dev())
                        if args.svd_modules in ['all', 'mlp', 'gaup', 'all_sep_fine', 'all_sep_fine_wo_down', 'all_sep']:
                            if args.mlp_fuse:
                                layer.mlp.up_proj.singular_values = singular_values_dict[layer_key]['up_proj'].to(utils.get_dev())
                            else:
                                layer.mlp.up_proj.singular_values = singular_values_dict[layer_key]['up_proj'].to(utils.get_dev())
                                layer.mlp.gate_proj.singular_values = singular_values_dict[layer_key]['gate_proj'].to(utils.get_dev())
                        if args.svd_modules in ['mlp', 'all', 'down', 'all_sep_fine', 'all_sep_fine_wo_down', 'all_sep']:
                            layer.mlp.down_proj.singular_values = singular_values_dict[layer_key]['down_proj'].to(utils.get_dev())
            logging.info("Successfully loaded singular values!")
        
        # Directly use pre-computed S gradient information
        singular_values_dict = {}
        device = utils.get_dev()  # Get CUDA device
        for idx, layer in enumerate(layers):
            layer_key = f"layer_{idx}"
            singular_values_dict[layer_key] = {}
            if args.svd_modules in ['qkv', 'attn', 'all']:
                if args.qkv_fuse:
                    raise NotImplementedError("QKV fuse is not supported for loss-aware singular value truncation")
                
                elif args.kv_fuse:
                    raise NotImplementedError("KV fuse is not supported for loss-aware singular value truncation")
                
                # elif args.inp_fuse:
                #     raise NotImplementedError("Inp fuse is not supported for loss-aware singular value truncation")

                else:
                    # Separate Q, K, V case (NEW)
                    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                        proj = getattr(layer.self_attn, proj_name)
                        if hasattr(proj, 'svd_info'):
                            svd_info = proj.svd_info
                            S = svd_info['S'].to(device).to(torch.float32)
                            
                            # Calculate importance score
                            singular_values_dict[layer_key][proj_name] = S.cpu()
                            print(f"Layer {idx} {proj_name} singular values collected, shape: {S.shape}")
                        else:
                            print(f"Warning: Layer {idx} {proj_name} lacks necessary singular values information")
            if args.svd_modules in ['attn', 'all', 'o']:
                svd_info = layer.self_attn.o_proj.svd_info
                S = svd_info['S'].to(device).to(torch.float32)
                
                singular_values_dict[layer_key]['o_proj'] = S.cpu()
                if args.is_rank_allocate_ft:
                    del layer.self_attn.o_proj.svd_info
                print(f"Layer {idx} O-proj singular values collected, shape: {S.shape}")

            if args.svd_modules in ['all', 'mlp', 'gaup']:
                if args.mlp_fuse:
                    svd_info = layer.mlp.up_proj.svd_info
                    S = svd_info['S'].to(device).to(torch.float32)
                    singular_values_dict[layer_key]['up_proj'] = S.cpu()
                    print(f"Layer {idx} Up-proj singular values collected, shape: {S.shape}")
                else:
                    svd_info = layer.mlp.up_proj.svd_info
                    S = svd_info['S'].to(device).to(torch.float32)
                    singular_values_dict[layer_key]['up_proj'] = S.cpu()
                    svd_info = layer.mlp.gate_proj.svd_info
                    S = svd_info['S'].to(device).to(torch.float32)
                    singular_values_dict[layer_key]['gate_proj'] = S.cpu()
                    print(f"Layer {idx} Gate-proj singular values collected, shape: {S.shape}")
                    if args.is_rank_allocate_ft:
                        del layer.mlp.gate_proj.svd_info
            if args.svd_modules in ['mlp', 'all', 'down']:
                svd_info = layer.mlp.down_proj.svd_info
                S = svd_info['S'].to(device).to(torch.float32)
                singular_values_dict[layer_key]['down_proj'] = S.cpu()
                print(f"Layer {idx} Down-proj singular values collected, shape: {S.shape}")
                if args.is_rank_allocate_ft:
                    del layer.mlp.down_proj.svd_info
                    

        # Save gradient importance score cache
        logging.info(f"Saving gradient importance score cache to {cache_file}...")
        torch.save(singular_values_dict, cache_file)
        logging.info("Singular values cache saved successfully!")
    
    selected_k_map = {}
    # Get indices and scores of top k important singular values
    num_layers = len(layers)
        
    if args.svd_modules == 'attn':
        if args.use_true_param_ratio:
            if args.is_per_head_svd:
                num_heads = layers[0].self_attn.config.num_key_value_heads if not args.is_q_headnum else layers[0].self_attn.config.num_attention_heads
                hidden_size =[(layers[0].self_attn.q_proj.weight.numel()/num_heads + layers[0].self_attn.k_proj.weight.numel()/num_heads  + layers[0].self_attn.v_proj.weight.numel()/num_heads)/ (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.q_proj.out_features/num_heads  + layers[0].self_attn.k_proj.out_features/num_heads + layers[0].self_attn.v_proj.out_features/num_heads ),\
                            layers[0].self_attn.o_proj.in_features * layers[0].self_attn.o_proj.out_features / (layers[0].self_attn.o_proj.in_features + layers[0].self_attn.o_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            else:
                hidden_size =[(layers[0].self_attn.q_proj.weight.numel() + layers[0].self_attn.k_proj.weight.numel() + layers[0].self_attn.v_proj.weight.numel())/ (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.q_proj.out_features + layers[0].self_attn.k_proj.out_features+ layers[0].self_attn.v_proj.out_features),\
                            layers[0].self_attn.o_proj.in_features * layers[0].self_attn.o_proj.out_features / (layers[0].self_attn.o_proj.in_features + layers[0].self_attn.o_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
        else:
            hidden_size = [layers[0].self_attn.k_proj.in_features, layers[0].self_attn.o_proj.in_features]
            total_rank = [num_layers * hsz for hsz in hidden_size]
            k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
        top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['k_proj','o_proj'], is_per_head_svd=args.is_per_head_svd)
    elif args.svd_modules == 'all':
        if args.mlp_fuse:
            if args.use_param_ratio:
                hidden_size =[layers[0].self_attn.k_proj.in_features * layers[0].self_attn.k_proj.out_features / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features), \
                    layers[0].self_attn.o_proj.in_features * layers[0].self_attn.o_proj.out_features / (layers[0].self_attn.o_proj.in_features + layers[0].self_attn.o_proj.out_features), \
                    layers[0].mlp.up_proj.in_features * layers[0].mlp.up_proj.out_features / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features), \
                    layers[0].mlp.down_proj.in_features * layers[0].mlp.down_proj.out_features / (layers[0].mlp.down_proj.in_features + layers[0].mlp.down_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            elif args.use_true_param_ratio:
                if args.is_per_head_svd:
                    logging.info("per head svd is not supported for all module for now")
                hidden_size =[(layers[0].self_attn.q_proj.weight.numel() + layers[0].self_attn.k_proj.weight.numel() + layers[0].self_attn.v_proj.weight.numel())/ (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.q_proj.out_features + layers[0].self_attn.k_proj.out_features+ layers[0].self_attn.v_proj.out_features), \
                    layers[0].self_attn.o_proj.in_features * layers[0].self_attn.o_proj.out_features / (layers[0].self_attn.o_proj.in_features + layers[0].self_attn.o_proj.out_features), \
                    layers[0].mlp.up_proj.weight.numel() * 2 / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features + layers[0].mlp.gate_proj.out_features), \
                    layers[0].mlp.down_proj.in_features * layers[0].mlp.down_proj.out_features / (layers[0].mlp.down_proj.in_features + layers[0].mlp.down_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            else:
                hidden_size = [layers[0].self_attn.k_proj.in_features, layers[0].self_attn.o_proj.in_features, layers[0].mlp.up_proj.in_features, layers[0].mlp.down_proj.out_features]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
            top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['k_proj','o_proj','up_proj','down_proj'])
        else:
            if args.use_param_ratio or args.use_true_param_ratio:
                hidden_size = [layers[0].self_attn.k_proj.in_features * layers[0].self_attn.k_proj.out_features / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features), \
                    layers[0].self_attn.o_proj.in_features * layers[0].self_attn.o_proj.out_features / (layers[0].self_attn.o_proj.in_features + layers[0].self_attn.o_proj.out_features), \
                    layers[0].mlp.up_proj.in_features * layers[0].mlp.up_proj.out_features / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features), \
                    layers[0].mlp.gate_proj.in_features * layers[0].mlp.gate_proj.out_features / (layers[0].mlp.gate_proj.in_features + layers[0].mlp.gate_proj.out_features), \
                    layers[0].mlp.down_proj.in_features * layers[0].mlp.down_proj.out_features / (layers[0].mlp.down_proj.in_features + layers[0].mlp.down_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            else:
                hidden_size = [layers[0].self_attn.k_proj.in_features, layers[0].self_attn.o_proj.in_features, layers[0].mlp.up_proj.in_features, layers[0].mlp.gate_proj.in_features, layers[0].mlp.down_proj.out_features]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
            top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['k_proj','o_proj','up_proj','gate_proj','down_proj'])
    elif args.svd_modules == 'mlp':
        if args.mlp_fuse:
            if args.use_param_ratio:
                hidden_size = [layers[0].mlp.up_proj.in_features * layers[0].mlp.up_proj.out_features / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features), \
                    layers[0].mlp.down_proj.in_features * layers[0].mlp.down_proj.out_features / (layers[0].mlp.down_proj.in_features + layers[0].mlp.down_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            elif args.use_true_param_ratio:
                hidden_size = [layers[0].mlp.up_proj.weight.numel() * 2 / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features + layers[0].mlp.gate_proj.out_features), \
                    layers[0].mlp.down_proj.in_features * layers[0].mlp.down_proj.out_features / (layers[0].mlp.down_proj.in_features + layers[0].mlp.down_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            else:
                hidden_size = [layers[0].mlp.up_proj.in_features, layers[0].mlp.down_proj.out_features]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
            top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['up_proj','down_proj'])
        else:
            if args.use_param_ratio or args.use_true_param_ratio:
                hidden_size = [layers[0].mlp.up_proj.in_features * layers[0].mlp.up_proj.out_features / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features), \
                    layers[0].mlp.gate_proj.in_features * layers[0].mlp.gate_proj.out_features / (layers[0].mlp.gate_proj.in_features + layers[0].mlp.gate_proj.out_features), \
                    layers[0].mlp.down_proj.in_features * layers[0].mlp.down_proj.out_features / (layers[0].mlp.down_proj.in_features + layers[0].mlp.down_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            else:
                hidden_size = [layers[0].mlp.up_proj.in_features, layers[0].mlp.gate_proj.in_features, layers[0].mlp.down_proj.out_features]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
            top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['up_proj','gate_proj','down_proj'])
    elif args.svd_modules == 'gaup':
        if args.mlp_fuse:
            if args.use_param_ratio:
                hidden_size = [layers[0].mlp.up_proj.in_features * layers[0].mlp.up_proj.out_features / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            elif args.use_true_param_ratio:
                hidden_size = [layers[0].mlp.up_proj.weight.numel() * 2 / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features + layers[0].mlp.gate_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            else:
                hidden_size = [layers[0].mlp.up_proj.in_features]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
            top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['up_proj'])
        else:
            if args.use_param_ratio:
                hidden_size = [layers[0].mlp.up_proj.in_features * layers[0].mlp.up_proj.out_features / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features), \
                    layers[0].mlp.gate_proj.in_features * layers[0].mlp.gate_proj.out_features / (layers[0].mlp.gate_proj.in_features + layers[0].mlp.gate_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            else:
                hidden_size = [layers[0].mlp.up_proj.in_features, layers[0].mlp.gate_proj.in_features]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
            top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['up_proj','gate_proj'])
    elif args.svd_modules == 'o':
        hidden_size = [layers[0].self_attn.o_proj.in_features]
        total_rank = [num_layers * hsz for hsz in hidden_size]
        k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
        top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['o_proj'])
    elif args.svd_modules == 'down':
        if args.use_param_ratio or args.use_true_param_ratio:
            hidden_size = [layers[0].mlp.down_proj.in_features * layers[0].mlp.down_proj.out_features / (layers[0].mlp.down_proj.in_features + layers[0].mlp.down_proj.out_features)]
            total_rank = [num_layers * hsz for hsz in hidden_size]
            k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
        else:
            hidden_size = [layers[0].mlp.down_proj.out_features]
            total_rank = [num_layers * hsz for hsz in hidden_size]
            k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
        top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['down_proj'])
    else: # qkv fuse
        if args.qkv_fuse:
            if args.use_true_param_ratio:
                if args.is_per_head_svd:
                    logging.info("added per head ratio compute here")
                    num_heads = layers[0].self_attn.config.num_key_value_heads  if not args.is_q_headnum else layers[0].self_attn.config.num_attention_heads
                    hidden_size =[(layers[0].self_attn.q_proj.weight.numel() + layers[0].self_attn.k_proj.weight.numel() + layers[0].self_attn.v_proj.weight.numel())/ (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.q_proj.out_features/num_heads + layers[0].self_attn.k_proj.out_features/num_heads + layers[0].self_attn.v_proj.out_features/num_heads)/num_heads]
                    total_rank = [num_layers * hsz for hsz in hidden_size]
                    k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
                else:
                    hidden_size =[(layers[0].self_attn.q_proj.weight.numel() + layers[0].self_attn.k_proj.weight.numel() + layers[0].self_attn.v_proj.weight.numel())/ (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.q_proj.out_features + layers[0].self_attn.k_proj.out_features+ layers[0].self_attn.v_proj.out_features)]
                    total_rank = [num_layers * hsz for hsz in hidden_size]
                    k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
                top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['k_proj'], is_per_head_svd=args.is_per_head_svd)
            elif args.use_all_true_param_ratio:
                if args.is_per_head_svd:
                    logging.info('adding all true p ratio then assign to each head')
                    num_heads = layers[0].self_attn.config.num_key_value_heads
                    hidden_size =[(layers[0].self_attn.q_proj.weight.numel() + layers[0].self_attn.k_proj.weight.numel() + layers[0].self_attn.v_proj.weight.numel())/ (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.q_proj.out_features + layers[0].self_attn.k_proj.out_features + layers[0].self_attn.v_proj.out_features)/num_heads]
                    total_rank = [num_layers * hsz for hsz in hidden_size]
                    k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
                else:
                    hidden_size =[(layers[0].self_attn.q_proj.weight.numel() + layers[0].self_attn.k_proj.weight.numel() + layers[0].self_attn.v_proj.weight.numel())/ (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.q_proj.out_features + layers[0].self_attn.k_proj.out_features+ layers[0].self_attn.v_proj.out_features)]
                    total_rank = [num_layers * hsz for hsz in hidden_size]
                    k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
                top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['k_proj'], is_per_head_svd=args.is_per_head_svd)

            else:
                hidden_size = layers[0].self_attn.k_proj.in_features
                total_rank = num_layers * hidden_size
                k_value = int(args.rank_ratio/2 * total_rank)
                top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=[k_value], keys=['k_proj'], is_per_head_svd=args.is_per_head_svd)
        
        elif args.kv_fuse:
            if args.use_true_param_ratio:
                if args.is_per_head_svd:
                    logging.info("per head ratio is not supported for kv fuse for now")
                # Calculate parameter ratios for Q and KV separately
                q_param_ratio = layers[0].self_attn.q_proj.weight.numel() / (layers[0].self_attn.q_proj.in_features + layers[0].self_attn.q_proj.out_features)
                kv_param_ratio = (layers[0].self_attn.k_proj.weight.numel() + layers[0].self_attn.v_proj.weight.numel()) / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features + layers[0].self_attn.v_proj.out_features)
                hidden_size = [q_param_ratio, kv_param_ratio]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
                top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['q_proj', 'k_proj'])
            else:
                hidden_size = layers[0].self_attn.k_proj.in_features
                total_rank = num_layers * hidden_size
                q_k_value = int(args.rank_ratio/2 * total_rank)  
                kv_k_value = int(args.rank_ratio/2 * total_rank)  
                k_value = [q_k_value, kv_k_value]
                top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['q_proj', 'k_proj'])
        # elif args.inp_fuse:
        #     if args.use_true_param_ratio:
        #         # Calculate parameter ratios for Q, K, V separately
        #         if args.is_per_head_svd:
        #             num_heads = layers[0].self_attn.config.num_key_value_heads
        #             num_headsq = layers[0].self_attn.config.num_attention_heads
        #             k_param_ratio = layers[0].self_attn.k_proj.weight.numel() * 3 / (layers[0].self_attn.k_proj.in_features * 3 + layers[0].self_attn.k_proj.out_features/num_heads)/num_heads
        #         else:
        #             k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features) # won't be okay for input concat.
        #         hidden_size = [k_param_ratio]
        #         total_rank = [num_layers * hsz for hsz in hidden_size]
        #         k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
        #         top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['k_proj'], is_per_head_svd=args.is_per_head_svd)
        #     # elif args.use_true_param_ratio: # _perhead
        #     #     # Calculate parameter ratios for Q, K, V separately
        #     #     if args.is_per_head_svd:
        #     #         num_heads = layers[0].self_attn.config.num_key_value_heads
        #     #         num_headsq = layers[0].self_attn.config.num_attention_heads
        #     #         k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features/num_heads)/num_heads
        #     #     else:
        #     #         k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features)
        #     #     hidden_size = [k_param_ratio]
        #     #     total_rank = [num_layers * hsz for hsz in hidden_size]
        #     #     k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
        #     #     top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['k_proj'], is_per_head_svd=args.is_per_head_svd)
        #     elif args.use_all_true_param_ratio:
        #         # Calculate parameter ratios for Q, K, V separately
        #         if args.is_per_head_svd:
        #             num_heads = layers[0].self_attn.config.num_key_value_heads
        #             k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features)/num_heads
        #         else:
        #             k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features)
        #         hidden_size = [k_param_ratio]
        #         total_rank = [num_layers * hsz for hsz in hidden_size]
        #         k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
        #         top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['k_proj'], is_per_head_svd=args.is_per_head_svd)
        #     else:
        #         hidden_size = layers[0].self_attn.k_proj.in_features
        #         total_rank = num_layers * hidden_size
        #         qkv_k_value = int(args.rank_ratio/3 * total_rank)
        #         k_value = [qkv_k_value]
        #         top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['k_proj'])
        # elif args.kvinp_fuse:
        #     if args.use_true_param_ratio:
        #         # Calculate parameter ratios for Q, K, V separately
        #         if args.is_per_head_svd:
        #             num_heads = layers[0].self_attn.config.num_key_value_heads
        #             num_headsq = layers[0].self_attn.config.num_attention_heads
        #             k_param_ratio = layers[0].self_attn.k_proj.weight.numel() * 2 / (layers[0].self_attn.k_proj.in_features * 2 + layers[0].self_attn.k_proj.out_features/num_heads)/num_heads
        #         else:
        #             k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features) # won't be okay for input concat.
        #         hidden_size = [k_param_ratio]
        #         total_rank = [num_layers * hsz for hsz in hidden_size]
        #         k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
        #         top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(singular_values_dict, k_list=k_value, keys=['k_proj'], is_per_head_svd=args.is_per_head_svd)
        else:  # Separate Q, K, V processing
            if args.use_true_param_ratio:
                # Calculate parameter ratios for Q, K, V separately
                if args.is_per_head_svd:
                    num_q_heads = layers[0].self_attn.config.num_attention_heads
                    num_heads = layers[0].self_attn.config.num_key_value_heads
                    q_param_ratio = layers[0].self_attn.q_proj.weight.numel() / (layers[0].self_attn.q_proj.in_features + layers[0].self_attn.q_proj.out_features/num_q_heads)/num_q_heads
                    k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features/num_heads)/num_heads
                    v_param_ratio = layers[0].self_attn.v_proj.weight.numel() / (layers[0].self_attn.v_proj.in_features + layers[0].self_attn.v_proj.out_features/num_heads)/num_heads
                else:
                    q_param_ratio = layers[0].self_attn.q_proj.weight.numel() / (layers[0].self_attn.q_proj.in_features + layers[0].self_attn.q_proj.out_features)
                    k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features)
                    v_param_ratio = layers[0].self_attn.v_proj.weight.numel() / (layers[0].self_attn.v_proj.in_features + layers[0].self_attn.v_proj.out_features)
                hidden_size = [q_param_ratio, k_param_ratio, v_param_ratio]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
                top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(
                    singular_values_dict,
                    k_list=k_value,
                    keys=['q_proj', 'k_proj', 'v_proj'],
                    is_per_head_svd=args.is_per_head_svd,
                    mixed_qkv_budget=getattr(args, "truncate_mixed_qkv_budget", False),
                )
            elif args.use_all_true_param_ratio:
                # Calculate parameter ratios for Q, K, V separately
                if args.is_per_head_svd:
                    num_q_heads = layers[0].self_attn.config.num_attention_heads
                    num_heads = layers[0].self_attn.config.num_key_value_heads
                    q_param_ratio = layers[0].self_attn.q_proj.weight.numel() / (layers[0].self_attn.q_proj.in_features + layers[0].self_attn.q_proj.out_features)/num_q_heads
                    k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features)/num_heads
                    v_param_ratio = layers[0].self_attn.v_proj.weight.numel() / (layers[0].self_attn.v_proj.in_features + layers[0].self_attn.v_proj.out_features)/num_heads
                else:
                    q_param_ratio = layers[0].self_attn.q_proj.weight.numel() / (layers[0].self_attn.q_proj.in_features + layers[0].self_attn.q_proj.out_features)
                    k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features)
                    v_param_ratio = layers[0].self_attn.v_proj.weight.numel() / (layers[0].self_attn.v_proj.in_features + layers[0].self_attn.v_proj.out_features)
                hidden_size = [q_param_ratio, k_param_ratio, v_param_ratio]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
                top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(
                    singular_values_dict,
                    k_list=k_value,
                    keys=['q_proj', 'k_proj', 'v_proj'],
                    is_per_head_svd=args.is_per_head_svd,
                    mixed_qkv_budget=getattr(args, "truncate_mixed_qkv_budget", False),
                )
            else:
                hidden_size = layers[0].self_attn.k_proj.in_features
                total_rank = num_layers * hidden_size
                qkv_k_value = int(args.rank_ratio/3 * total_rank)
                k_value = [qkv_k_value, qkv_k_value, qkv_k_value]
                top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(
                    singular_values_dict,
                    k_list=k_value,
                    keys=['q_proj', 'k_proj', 'v_proj'],
                    is_per_head_svd=args.is_per_head_svd,
                    mixed_qkv_budget=getattr(args, "truncate_mixed_qkv_budget", False),
                )
    
    pre_truncate_snapshot = _selection_budget_snapshot(layer_indices_dict)
    floor_meta = {}
    if bool(getattr(args, "truncate_hybrid_grad_layer_budget", False)):
        _, _, grad_layer_indices_dict = grad_info_utils.svd_qkv_with_grad_info(
            layers,
            args,
            use_cache=getattr(args, "use_cache", True),
        )
        layer_indices_dict, hybrid_meta = apply_grad_layer_budget_hybrid(
            layer_indices_dict=layer_indices_dict,
            singular_values_dict=singular_values_dict,
            grad_layer_indices_dict=grad_layer_indices_dict,
            args=args,
            num_layers=num_layers,
        )
        floor_meta.update(hybrid_meta)
    elif bool(getattr(args, "truncate_hybrid_loss_aware_layer_budget", False)):
        layer_indices_dict, hybrid_meta = apply_loss_aware_layer_budget_hybrid(
            layer_indices_dict=layer_indices_dict,
            singular_values_dict=singular_values_dict,
            args=args,
            num_layers=num_layers,
        )
        floor_meta.update(hybrid_meta)

    if bool(getattr(args, "truncate_balance_penalty", False)):
        layer_indices_dict, balance_meta = apply_per_head_balance_penalty(
            layer_indices_dict=layer_indices_dict,
            singular_values_dict=singular_values_dict,
            args=args,
            num_layers=num_layers,
        )
        floor_meta.update(balance_meta)
    else:
        layer_indices_dict, hard_floor_meta = apply_per_head_rank_floor(
            layer_indices_dict=layer_indices_dict,
            singular_values_dict=singular_values_dict,
            args=args,
            num_layers=num_layers,
        )
        floor_meta.update(hard_floor_meta)
    top_indices, top_scores = rebuild_topk_from_layer_indices(
        layer_indices_dict=layer_indices_dict,
        singular_values_dict=singular_values_dict,
        is_per_head_svd=getattr(args, "is_per_head_svd", False),
    )

    if getattr(args, "act_calib_debug", False):
        post_truncate_snapshot = _selection_budget_snapshot(layer_indices_dict)
        selection_debug_payload = {}
        for layer_idx, proj_map in layer_indices_dict.items():
            layer_key = f"layer_{layer_idx}"
            for proj_name, selected_indices in proj_map.items():
                module_name = (
                    f"model.layers.{layer_idx}.self_attn.{proj_name}"
                    if proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]
                    else f"model.layers.{layer_idx}.mlp.{proj_name}"
                )
                entry = selection_debug_payload.setdefault(module_name, {})
                S_cpu = singular_values_dict.get(layer_key, {}).get(proj_name)
                available_count = None
                selected_scores = None
                if isinstance(S_cpu, torch.Tensor):
                    available_count = int(S_cpu.shape[-1] if S_cpu.ndim >= 1 else S_cpu.numel())
                    if isinstance(selected_indices, dict) and S_cpu.ndim >= 2:
                        gathered = []
                        for head_idx in sorted(selected_indices):
                            idxs = list(selected_indices[head_idx])
                            if idxs:
                                gathered.append(S_cpu[int(head_idx), idxs].reshape(-1))
                        if gathered:
                            selected_scores = torch.cat(gathered, dim=0)
                    else:
                        idxs = list(selected_indices or [])
                        if idxs:
                            selected_scores = S_cpu[idxs]
                debug_utils.attach_selection_debug(
                    entry,
                    selected_indices=selected_indices,
                    available_count=available_count,
                    allocated_rank_ratio=(None if available_count is None else (sum(len(v) for v in selected_indices.values()) if isinstance(selected_indices, dict) else len(selected_indices)) / max(available_count, 1)),
                    selected_scores=selected_scores,
                )
                floor_info = floor_meta.get(module_name)
                debug_utils.attach_floor_debug(
                    entry,
                    floor_enabled=bool(floor_info),
                    floor_target=(floor_info or {}).get("floor_target"),
                    floor_before_head_count=(floor_info or {}).get("floor_before_head_count"),
                    floor_after_head_count=(floor_info or {}).get("floor_after_head_count"),
                    floor_added_count=(floor_info or {}).get("floor_added_count"),
                    floor_removed_count=(floor_info or {}).get("floor_removed_count"),
                    floor_unsatisfied=(floor_info or {}).get("floor_unsatisfied"),
                    floor_cross_layer_enabled=(floor_info or {}).get("floor_cross_layer_enabled"),
                )
                pre_stage = pre_truncate_snapshot.get(int(layer_idx), {}).get(proj_name, {})
                debug_utils.attach_budget_stage_debug(
                    entry,
                    stage_name="pre_truncate",
                    layer_budget=pre_stage.get("layer_budget"),
                    head_budget=pre_stage.get("head_budget"),
                    total_budget=pre_stage.get("total_budget"),
                )
                post_stage = post_truncate_snapshot.get(int(layer_idx), {}).get(proj_name, {})
                debug_utils.attach_budget_stage_debug(
                    entry,
                    stage_name="post_truncate",
                    layer_budget=post_stage.get("layer_budget"),
                    head_budget=post_stage.get("head_budget"),
                    total_budget=post_stage.get("total_budget"),
                )
                requested = selected_k_map.get(proj_name)
                if requested is not None:
                    entry["requested_topk"] = int(requested if not isinstance(requested, list) else requested[0])
        debug_utils.save_act_calib_debug_payload(
            args,
            filename="act_calib_debug_svd_selection",
            payload=selection_debug_payload,
            log_prefix="act_calib_debug (selection)",
        )
    for key, indices in top_indices.items():
        logging.info(f"Selected top {len(indices)} important singular values for {key}")
    return top_indices, top_scores, layer_indices_dict

def prepare_svd(model, args):
    """
    Pre-process QKV layers with SVD decomposition and store results in attention modules
    
    Args:
        model: Model to be processed
        args: Parameter configuration
    """
    print("Preprocessing QKV layer SVD decomposition...")
    device = utils.get_dev()
    alpha = args.act_alpha
    svd_debug_payload = {} if getattr(args, "act_calib_debug", False) else None
    
    for idx, layer in enumerate(tqdm(model_utils.get_layers(model), desc="Preparing QKV SVD")):
        if args.svd_modules in ['qkv', 'attn', 'all']:
            q_linear = layer.self_attn.q_proj
            k_linear = layer.self_attn.k_proj
            v_linear = layer.self_attn.v_proj
            if args.qkv_fuse:
                raise NotImplementedError("QKV fuse is not supported for loss-aware SVD")
            elif args.kv_fuse:
                raise NotImplementedError("KV fuse is not supported for loss-aware SVD")
            # elif args.inp_fuse:
            #     raise NotImplementedError("Inp fuse is not supported for loss-aware SVD")
            else:
                # Handle separate Q, K, V
                for proj_name, proj_layer in [('q_proj', q_linear), ('k_proj', k_linear), ('v_proj', v_linear)]:
                    try:
                        if args.is_per_head_svd:
                            if proj_name == 'q_proj':
                                num_heads = layer.self_attn.config.num_attention_heads
                            else:
                                num_heads = layer.self_attn.config.num_key_value_heads
                            w = proj_layer.weight.data.view(num_heads, -1, proj_layer.in_features).float().to(utils.get_dev())
                        else:
                            w = proj_layer.weight.data.float().to(utils.get_dev())
                        w_before = w.clone()
                        input_whiten_applied = False
                        input_whiten_is_full = False
                        if args.act_aware or args.loss_aware:
                            scaling_diag_matrix = torch.ones(proj_layer.in_features, device=device)
                            if hasattr(k_linear, "scaling_diag_matrix") and k_linear.scaling_diag_matrix is not None:
                                scaling_diag_matrix *= k_linear.scaling_diag_matrix.to(device)**alpha
                                scaling_diag_matrix += 1e-6
                                scaling_matrix_inverse_transpose = scaling_diag_matrix**(-1)
                                input_whiten_applied = True
                                if args.is_per_head_svd:
                                    w = w * scaling_diag_matrix.view(1, 1, -1)
                                else:
                                    w = w * scaling_diag_matrix.view(1, -1)
                            elif hasattr(k_linear, "scaling_diag_matrixS") and k_linear.scaling_diag_matrixS is not None:
                                scaling_diag_matrix = k_linear.scaling_diag_matrixS.to(device)
                                try:
                                    scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                                except Exception as e:
                                    logging.info("Warning: scaling_diag_matrix is not full rank!")
                                    scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                                    scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                                input_whiten_applied = True
                                input_whiten_is_full = True
                                w = w @ scaling_diag_matrix.float()
                        
                        if args.loss_aware:
                            output_grad_scaling_matrix = proj_layer.output_grad_scaling_matrix.to(device)
                            w = output_grad_scaling_matrix.float() @ w
                            output_grad_scaling_matrix_inv = torch.linalg.inv(output_grad_scaling_matrix.float())
                        else:
                            output_grad_scaling_matrix_inv = None
                        whitened_w = w.clone()
                            
                        U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False)
                        if args.is_per_head_svd:
                            V = Vt.transpose(1, 2)
                        else:
                            V = Vt.T
                            
                        if args.loss_aware:
                            U = output_grad_scaling_matrix_inv.float() @ U
                        
                        if input_whiten_applied:
                            if input_whiten_is_full:
                                if args.is_per_head_svd:
                                    V = scaling_matrix_inverse_transpose.float().unsqueeze(0).expand(V.shape[0], -1, -1) @ V
                                else:
                                    V = scaling_matrix_inverse_transpose.float() @ V
                            else:
                                if args.is_per_head_svd:
                                    V = V / scaling_diag_matrix.view(1, -1, 1)
                                else:
                                    V = V / scaling_diag_matrix.view(1, -1)
                        
                        proj_layer.svd_info = {
                            'U': U.cpu(),
                            'S': S.cpu(),
                            'V': V.cpu()
                        }
                        debug_utils.record_prepare_svd_debug(
                            svd_debug_payload,
                            key=f"model.layers.{idx}.self_attn.{proj_name}",
                            weight_before=w_before,
                            weight_after=whitened_w,
                            S=S,
                            output_grad_scaling_matrix_inv=output_grad_scaling_matrix_inv,
                            input_inverse=scaling_matrix_inverse_transpose if input_whiten_applied else None,
                        )
                        print(f"Layer {idx} {proj_name} SVD completed, S shape: {S.shape}")
                    except Exception as e:
                        print(f"Layer {idx} {proj_name} SVD failed: {e}")
                        import traceback
                        traceback.print_exc()
        if args.svd_modules in ['attn', 'all', 'o']:
            o_proj = layer.self_attn.o_proj
            try:
                # Move weights to CUDA device
                w = o_proj.weight.data.float().to(utils.get_dev())
                w_before = w.clone()

                # Apply activation-aware scaling (if enabled)
                input_whiten_applied = False
                if args.act_aware or args.loss_aware:
                    scaling_diag_matrix = torch.ones(o_proj.in_features, device=utils.get_dev())  # avoid zero division
                    if hasattr(o_proj, "scaling_diag_matrix"):
                        scaling_diag_matrix *= o_proj.scaling_diag_matrix.to(utils.get_dev())**alpha
                        scaling_diag_matrix += 1e-6  # avoid zero division
                        scaling_matrix_inverse_transpose = scaling_diag_matrix**(-1)
                        input_whiten_applied = True
                        w = w * scaling_diag_matrix.view(1, -1)
                    elif hasattr(o_proj, "scaling_diag_matrixS"):
                        scaling_diag_matrix = o_proj.scaling_diag_matrixS.to(utils.get_dev())
                        try:
                            scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                        except Exception as e:
                            logging.info("Warning: scaling_diag_matrix is not full rank!")
                            scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                            scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                        input_whiten_applied = True
                        w = w @ scaling_diag_matrix.float()
                    else:
                        logging.warning("No scaling_diag_matrix found for o_proj, skip input whitening.")
                
                if args.loss_aware:
                    output_grad_scaling_matrix = o_proj.output_grad_scaling_matrix.to(device)
                    w = output_grad_scaling_matrix.float() @ w
                    output_grad_scaling_matrix_inv = torch.linalg.inv(output_grad_scaling_matrix.float())
                else:
                    output_grad_scaling_matrix_inv = None
                whitened_w = w.clone()
                    
                # SVD decomposition
                U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False) # SVD decomposition of WS
                V = Vt.T
                
                if args.loss_aware:
                    U = output_grad_scaling_matrix_inv.float() @ U
                    
                if input_whiten_applied:
                    if hasattr(o_proj, "scaling_diag_matrix"):
                        V = V / scaling_diag_matrix.view(1, -1)
                    elif hasattr(o_proj, "scaling_diag_matrixS"):
                        V = scaling_matrix_inverse_transpose.float() @ V
                        
                # Store SVD results
                layer.self_attn.o_proj.svd_info = {
                    'U': U.cpu(),
                    'S': S.cpu(),
                    'V': V.cpu()
                }
                debug_utils.record_prepare_svd_debug(
                    svd_debug_payload,
                    key=f"model.layers.{idx}.self_attn.o_proj",
                    weight_before=w_before,
                    weight_after=whitened_w,
                    S=S,
                    output_grad_scaling_matrix_inv=output_grad_scaling_matrix_inv,
                    input_inverse=scaling_matrix_inverse_transpose if input_whiten_applied else None,
                )
                print(f"Layer {idx} O-proj SVD completed, S shape: {S.shape}")
            except Exception as e:
                print(f"Layer {idx} O-proj SVD failed: {e}")
                import traceback
                traceback.print_exc()

        if args.svd_modules in ['all', 'mlp', 'gaup']:
            up_proj = layer.mlp.up_proj
            gate_proj = layer.mlp.gate_proj
            if args.mlp_fuse:
                try:
                    # Move weights to CUDA device
                    w = torch.cat([
                        up_proj.weight.data.float(),
                        gate_proj.weight.data.float()
                    ], dim=0).to(utils.get_dev())
                    w_before = w.clone()
                    
                    # Apply activation-aware scaling (if enabled)
                    input_whiten_applied = False
                    if args.act_aware or args.loss_aware:
                        scaling_diag_matrix = torch.ones(up_proj.in_features, device=utils.get_dev())  # avoid zero division
                        if hasattr(up_proj, "scaling_diag_matrix"):
                            scaling_diag_matrix *= up_proj.scaling_diag_matrix.to(utils.get_dev())**alpha
                            scaling_diag_matrix += 1e-6  # avoid zero division
                            scaling_matrix_inverse_transpose = scaling_diag_matrix**(-1)
                            input_whiten_applied = True
                            w = w * scaling_diag_matrix.view(1, -1)
                        elif hasattr(up_proj, "scaling_diag_matrixS"):
                            scaling_diag_matrix = up_proj.scaling_diag_matrixS.to(utils.get_dev())
                            try:
                                scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                            except Exception as e:
                                logging.info("Warning: scaling_diag_matrix is not full rank!")
                                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                                scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                            input_whiten_applied = True
                            w = w @ scaling_diag_matrix.float()
                    
                    if args.loss_aware:
                        output_grad_scaling_matrix = up_proj.output_grad_scaling_matrix.to(device)
                        w = output_grad_scaling_matrix.float() @ w
                        output_grad_scaling_matrix_inv = torch.linalg.inv(output_grad_scaling_matrix.float())
                    else:
                        output_grad_scaling_matrix_inv = None
                    whitened_w = w.clone()
                        
                    # SVD decomposition
                    U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False) # SVD decomposition of WS
                    V = Vt.T
                    
                    if args.loss_aware:
                        U = output_grad_scaling_matrix_inv.float() @ U
                        
                    if input_whiten_applied:
                        if hasattr(up_proj, "scaling_diag_matrix"):
                            V = V / scaling_diag_matrix.view(1, -1)
                        elif hasattr(up_proj, "scaling_diag_matrixS"):
                            V = scaling_matrix_inverse_transpose.float() @ V
                            
                    # Store SVD results
                    layer.mlp.up_proj.svd_info = {
                        'U': U.cpu(),
                        'S': S.cpu(),
                        'V': V.cpu()
                    }
                    debug_utils.record_prepare_svd_debug(
                        svd_debug_payload,
                        key=f"model.layers.{idx}.mlp.up_proj_fused",
                        weight_before=w_before,
                        weight_after=whitened_w,
                        S=S,
                        output_grad_scaling_matrix_inv=output_grad_scaling_matrix_inv,
                        input_inverse=scaling_matrix_inverse_transpose if input_whiten_applied else None,
                    )
                    print(f"Layer {idx} Up-proj SVD completed, S shape: {S.shape}")
                except Exception as e:
                    print(f"Layer {idx} Up-proj SVD failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                try:
                    w = up_proj.weight.data.float().to(utils.get_dev())
                    w_before = w.clone()
                    input_whiten_applied = False
                    if args.act_aware or args.loss_aware:
                        scaling_diag_matrix = torch.ones(up_proj.in_features, device=utils.get_dev())  # avoid zero division
                        if hasattr(up_proj, "scaling_diag_matrix"):
                            scaling_diag_matrix *= up_proj.scaling_diag_matrix.to(utils.get_dev())**alpha
                            scaling_diag_matrix += 1e-6  # avoid zero division
                            scaling_matrix_inverse_transpose = scaling_diag_matrix**(-1)
                            input_whiten_applied = True
                            w = w * scaling_diag_matrix.view(1, -1)
                        elif hasattr(up_proj, "scaling_diag_matrixS"):
                            scaling_diag_matrix = up_proj.scaling_diag_matrixS.to(utils.get_dev())
                            try:
                                scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                            except Exception as e:
                                logging.info("Warning: scaling_diag_matrix is not full rank!")
                                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                                scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                            input_whiten_applied = True
                            w = w @ scaling_diag_matrix.float()
                    
                    if args.loss_aware:
                        output_grad_scaling_matrix = up_proj.output_grad_scaling_matrix.to(device)
                        w = output_grad_scaling_matrix.float() @ w
                        output_grad_scaling_matrix_inv = torch.linalg.inv(output_grad_scaling_matrix.float())
                    else:
                        output_grad_scaling_matrix_inv = None
                    whitened_w = w.clone()
                        
                    U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False) # SVD decomposition of WS
                    V = Vt.T
                    
                    if args.loss_aware:
                        U = output_grad_scaling_matrix_inv.float() @ U
                        
                    if input_whiten_applied:
                        if hasattr(up_proj, "scaling_diag_matrix"):
                            V = V / scaling_diag_matrix.view(1, -1)
                        elif hasattr(up_proj, "scaling_diag_matrixS"):
                            V = scaling_matrix_inverse_transpose.float() @ V
                            
                    layer.mlp.up_proj.svd_info = {
                        'U': U.cpu(),
                        'S': S.cpu(),
                        'V': V.cpu()
                    }
                    debug_utils.record_prepare_svd_debug(
                        svd_debug_payload,
                        key=f"model.layers.{idx}.mlp.up_proj",
                        weight_before=w_before,
                        weight_after=whitened_w,
                        S=S,
                        output_grad_scaling_matrix_inv=output_grad_scaling_matrix_inv,
                        input_inverse=scaling_matrix_inverse_transpose if input_whiten_applied else None,
                    )
                    print(f"Layer {idx} Up-proj SVD completed, S shape: {S.shape}")
                except Exception as e:
                    print(f"Layer {idx} Up-proj SVD failed: {e}")
                    import traceback
                    traceback.print_exc()
                try:
                    w = gate_proj.weight.data.float().to(utils.get_dev())
                    w_before = w.clone()
                    input_whiten_applied = False
                    if args.act_aware or args.loss_aware:
                        scaling_diag_matrix = torch.ones(up_proj.in_features, device=utils.get_dev())  # avoid zero division
                        if hasattr(up_proj, "scaling_diag_matrix"):
                            scaling_diag_matrix *= up_proj.scaling_diag_matrix.to(utils.get_dev())**alpha
                            scaling_diag_matrix += 1e-6  # avoid zero division
                            scaling_matrix_inverse_transpose = scaling_diag_matrix**(-1)
                            input_whiten_applied = True
                            w = w * scaling_diag_matrix.view(1, -1)
                        elif hasattr(up_proj, "scaling_diag_matrixS"):
                            scaling_diag_matrix = up_proj.scaling_diag_matrixS.to(utils.get_dev())
                            try:
                                scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                            except Exception as e:
                                logging.info("Warning: scaling_diag_matrix is not full rank!")
                                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                                scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                            input_whiten_applied = True
                            w = w @ scaling_diag_matrix.float()
                            
                    if args.loss_aware:
                        output_grad_scaling_matrix = up_proj.output_grad_scaling_matrix.to(device)
                        w = output_grad_scaling_matrix.float() @ w
                        output_grad_scaling_matrix_inv = torch.linalg.inv(output_grad_scaling_matrix.float())
                    else:
                        output_grad_scaling_matrix_inv = None
                    whitened_w = w.clone()
                        
                    U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False) # SVD decomposition of WS
                    V = Vt.T
                    
                    if args.loss_aware:
                        U = output_grad_scaling_matrix_inv.float() @ U
                        
                    if input_whiten_applied:
                        if hasattr(up_proj, "scaling_diag_matrix"):
                            V = V / scaling_diag_matrix.view(1, -1)
                        elif hasattr(up_proj, "scaling_diag_matrixS"):
                            V = scaling_matrix_inverse_transpose.float() @ V
                            
                    layer.mlp.gate_proj.svd_info = {
                        'U': U.cpu(),
                        'S': S.cpu(),
                        'V': V.cpu()
                    }
                    debug_utils.record_prepare_svd_debug(
                        svd_debug_payload,
                        key=f"model.layers.{idx}.mlp.gate_proj",
                        weight_before=w_before,
                        weight_after=whitened_w,
                        S=S,
                        output_grad_scaling_matrix_inv=output_grad_scaling_matrix_inv,
                        input_inverse=scaling_matrix_inverse_transpose if input_whiten_applied else None,
                    )
                    print(f"Layer {idx} Gate-proj SVD completed, S shape: {S.shape}")
                except Exception as e:
                    print(f"Layer {idx} Gate-proj SVD failed: {e}")
                    import traceback
                    traceback.print_exc()

        if args.svd_modules in ['all', 'mlp', 'down']:
            down_proj = layer.mlp.down_proj
            try:
                w = down_proj.weight.data.float().to(utils.get_dev())
                w_before = w.clone()

                input_whiten_applied = False
                if args.act_aware or args.loss_aware:
                    scaling_diag_matrix = torch.ones(down_proj.in_features, device=utils.get_dev())  # avoid zero division
                    if hasattr(down_proj, "scaling_diag_matrix"):
                        scaling_diag_matrix *= down_proj.scaling_diag_matrix.to(utils.get_dev())**alpha
                        scaling_diag_matrix += 1e-6  # avoid zero division 
                        scaling_matrix_inverse_transpose = scaling_diag_matrix**(-1)
                        input_whiten_applied = True
                        w = w * scaling_diag_matrix.view(1, -1)
                    elif hasattr(down_proj, "scaling_diag_matrixS"):
                        scaling_diag_matrix = down_proj.scaling_diag_matrixS.to(utils.get_dev())
                        try:
                            scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                        except Exception as e:
                            logging.info("Warning: scaling_diag_matrix is not full rank!")
                            scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                            scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                        input_whiten_applied = True
                        w = w @ scaling_diag_matrix.float()
                    
                    if args.loss_aware:
                        output_grad_scaling_matrix = down_proj.output_grad_scaling_matrix.to(device)
                        w = output_grad_scaling_matrix.float() @ w
                        output_grad_scaling_matrix_inv = torch.linalg.inv(output_grad_scaling_matrix.float())
                    else:
                        output_grad_scaling_matrix_inv = None
                    whitened_w = w.clone()
                        
                    U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False) # SVD decomposition of WS
                    V = Vt.T
                    
                    if args.loss_aware:
                        U = output_grad_scaling_matrix_inv.float() @ U
                        
                    if input_whiten_applied:
                        if hasattr(down_proj, "scaling_diag_matrix"):
                            V = V / scaling_diag_matrix.view(1, -1)
                        elif hasattr(down_proj, "scaling_diag_matrixS"):
                            V = scaling_matrix_inverse_transpose.float() @ V
                            
                    layer.mlp.down_proj.svd_info = {
                        'U': U.cpu(),
                        'S': S.cpu(),
                        'V': V.cpu()
                    }
                    debug_utils.record_prepare_svd_debug(
                        svd_debug_payload,
                        key=f"model.layers.{idx}.mlp.down_proj",
                        weight_before=w_before,
                        weight_after=whitened_w,
                        S=S,
                        output_grad_scaling_matrix_inv=output_grad_scaling_matrix_inv,
                        input_inverse=scaling_matrix_inverse_transpose if input_whiten_applied else None,
                    )
                    print(f"Layer {idx} Down-proj SVD completed, S shape: {S.shape}")
            except Exception as e:
                print(f"Layer {idx} Down-proj SVD failed: {e}")
                import traceback
                traceback.print_exc()
    
    if args.svd_modules in ['qkv', 'attn', 'all']:
        print("QKV layer SVD preprocessing completed")
    if args.svd_modules in ['attn', 'all', 'o']:
        print("O-proj layer SVD preprocessing completed")
    if args.svd_modules in ['all', 'mlp', 'gaup', 'down']:
        print("MLP layer SVD preprocessing completed")
    if svd_debug_payload is not None:
        debug_utils.save_act_calib_debug_payload(
            args,
            filename="act_calib_debug_svd",
            payload=svd_debug_payload,
            log_prefix="act_calib_debug",
        )


def quick_loss_aware_numerical_consistency_check(
    out_dim=16,
    in_dim=12,
    rank=6,
    damping=1e-3,
    seed=0,
    device=None,
    dtype=torch.float64,
    atol=1e-7,
    rtol=1e-5,
):
    """
    Quick numerical sanity-check for loss-aware SVD theory:
      1) Full-rank reconstruction recovers W from M = P W Q.
      2) Truncated reconstruction satisfies:
         ||P(W - W_hat)Q||_F^2 == ||M - M_hat||_F^2 == sum(discarded singular values^2)

    Uses random small matrices only (no model dependency).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    # Random weight matrix W \in R^{out_dim x in_dim}
    W = torch.randn(out_dim, in_dim, generator=g, device=device, dtype=dtype)

    # Build SPD covariances:
    #   P^T P = G  -> use upper Cholesky
    #   Q Q^T = A  -> use lower Cholesky
    G0 = torch.randn(out_dim, out_dim, generator=g, device=device, dtype=dtype)
    A0 = torch.randn(in_dim, in_dim, generator=g, device=device, dtype=dtype)
    eye_out = torch.eye(out_dim, device=device, dtype=dtype)
    eye_in = torch.eye(in_dim, device=device, dtype=dtype)
    G = G0 @ G0.transpose(-1, -2) + float(damping) * eye_out
    A = A0 @ A0.transpose(-1, -2) + float(damping) * eye_in

    P = torch.linalg.cholesky(G, upper=True)   # P^T P = G
    Q = torch.linalg.cholesky(A, upper=False)  # Q Q^T = A

    # Whitened matrix and its SVD
    M = P @ W @ Q
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    min_dim = S.numel()
    rank = max(0, min(int(rank), int(min_dim)))

    # Full-rank reconstruction should recover W
    M_full = U @ torch.diag(S) @ Vt
    W_full = torch.linalg.solve(P, M_full) @ torch.linalg.inv(Q)

    # Truncated reconstruction
    if rank == 0:
        M_trunc = torch.zeros_like(M)
    else:
        M_trunc = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]
    W_trunc = torch.linalg.solve(P, M_trunc) @ torch.linalg.inv(Q)

    # Theoretical identities
    lhs = torch.linalg.norm(P @ (W - W_trunc) @ Q, ord="fro") ** 2
    rhs_residual = torch.linalg.norm(M - M_trunc, ord="fro") ** 2
    rhs_sigma = torch.sum(S[rank:] ** 2)

    full_recon_ok = torch.allclose(W_full, W, rtol=rtol, atol=atol)
    metric_ok = (
        torch.allclose(lhs, rhs_residual, rtol=rtol, atol=atol)
        and torch.allclose(lhs, rhs_sigma, rtol=rtol, atol=atol)
    )

    return {
        "pass": bool(full_recon_ok and metric_ok),
        "full_recon_ok": bool(full_recon_ok),
        "metric_ok": bool(metric_ok),
        "full_recon_rel_error": float((torch.linalg.norm(W_full - W) / (torch.linalg.norm(W) + 1e-18)).item()),
        "lhs_weighted_residual": float(lhs.item()),
        "rhs_whitened_residual": float(rhs_residual.item()),
        "rhs_discarded_sigma_sq": float(rhs_sigma.item()),
        "rank": int(rank),
        "shape": [int(out_dim), int(in_dim)],
        "dtype": str(dtype),
        "device": str(device),
        "atol": float(atol),
        "rtol": float(rtol),
    }
