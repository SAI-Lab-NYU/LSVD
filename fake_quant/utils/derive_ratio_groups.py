"""
derive_ratio_groups.py
----------------------
Derive hybrid-FFN channel-group JSON files for multiple svd_ffn_ratio values
from a single Phase-1 svd_aware JSON that contains `greedy_accept_order`.

Because the greedy algorithm processes candidates in a fixed order independent
of the target ratio, the first K entries of `greedy_accept_order` are exactly
the channels that would be assigned to Quant for any ratio whose n_quant = K.
So we only need ONE Phase-1 run (with any ratio) to evaluate all ratios.

Usage
-----
    python derive_ratio_groups.py \\
        --src  $LSVD_GROUPS_DIR/hybrid_groups_llava_rtn_w8_r02_svd_aware.json \\
        --out  $LSVD_GROUPS_DIR/ \\
        --ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
"""

import argparse
import json
import os
import copy

def derive(src_path: str, out_dir: str, ratios: list[float], prefix: str = "hybrid_groups_llava_rtn_w8") -> list[str]:
    with open(src_path) as f:
        src = json.load(f)

    # Verify the source JSON has greedy_accept_order
    first_layer = src["layers"]["0"]
    if "greedy_accept_order" not in first_layer:
        raise ValueError(
            "Source JSON does not contain 'greedy_accept_order'. "
            "Re-run Phase 1 with the updated hybrid_ffn_utils.py."
        )

    d_ffn     = first_layer["d_ffn"]
    n_layers  = len(src["layers"])
    out_paths = []

    for ratio in ratios:
        n_svd   = max(1, round(ratio * d_ffn))
        n_quant = d_ffn - n_svd

        out = copy.deepcopy(src)
        out["meta"]["svd_ffn_ratio"] = ratio

        for li in range(n_layers):
            layer = out["layers"][str(li)]
            order = layer["greedy_accept_order"]  # full acceptance order

            # Take first n_quant accepted channels as the quant group
            quant_idx = sorted(order[:n_quant])
            svd_idx   = sorted(set(range(d_ffn)) - set(quant_idx))
            perm      = svd_idx + quant_idx      # svd first, then quant

            layer["svd_idx"]   = svd_idx
            layer["quant_idx"] = quant_idx
            layer["perm"]      = perm
            layer["n_svd"]     = len(svd_idx)
            # Keep greedy_accept_order so further ratios can be derived later

        ratio_str = f"{int(ratio*100):02d}"
        fname     = f"{prefix}_r{ratio_str}_svd_aware.json"
        out_path  = os.path.join(out_dir, fname)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[derive] ratio={ratio:.1f}  SVD={n_svd}  Quant={n_quant}  → {out_path}")
        out_paths.append(out_path)

    return out_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",    required=True, help="Source svd_aware JSON from Phase 1")
    parser.add_argument("--out",    required=True, help="Output directory for derived JSONs")
    parser.add_argument("--ratios", nargs="+", type=float,
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--prefix", type=str, default="hybrid_groups_llava_rtn_w8",
                        help="Output filename prefix")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = derive(args.src, args.out, args.ratios, prefix=args.prefix)
    print(f"\nGenerated {len(paths)} JSON files in {args.out}")
