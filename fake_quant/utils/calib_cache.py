"""Released calibration caches (--calib_cache_dir).

The LSVD release ships, per model, the four W8A8 calibration statistics under fixed file names, plus a
calib_cache.json that records the calibration settings they were computed with. With --calib_cache_dir
these files are read instead of the shared calibration cache (experiments/<model>/<namespace>/cache/,
keyed by nsamples and seed) and are never recomputed or written: a missing file, or a run whose
calibration settings differ from calib_cache.json, is an error.
"""
import json
import logging
import os

MANIFEST = "calib_cache.json"
FILES = {
    "act_input": "act_input.pt",  # Q/K/V input-side Cholesky factors (act_aware_utils)
    "output_grad": "output_grad.pt",  # per-head output-gradient Cholesky factors (loss_aware_utils)
    "grad_info": "grad_info.pt",  # gradient (Fisher) information per head (grad_info_utils)
    "grad_scores": "grad_scores.pt",  # importance scores for rank allocation (grad_info_utils)
}


def run_config(args):
    """The arguments that select the calibration statistics (the tokens of the shared cache file names)."""
    a_bits_calib = args.a_bits_calib if args.a_bits_calib is not None else args.a_bits
    return {
        "model": args.model,
        "cal_dataset": args.cal_dataset,
        "nsamples": args.nsamples,
        "calib_method": args.calib_method,
        "act_calib_fake_quant": bool(args.act_calib_fake_quant),
        "a_bits_calib": a_bits_calib,
        "a_groupsize": args.a_groupsize,
        "a_asym": bool(args.a_asym),
        "a_clip_ratio": args.a_clip_ratio,
        "svd_modules": args.svd_modules,
        "is_per_head_svd": bool(args.is_per_head_svd),
        "qkv_fuse": bool(args.qkv_fuse),
        "kv_fuse": bool(args.kv_fuse),
        "taylor_order": args.taylor_order,
        "grad_alpha": args.grad_alpha,
        "pre_svd_rotate": bool(getattr(args, "pre_svd_rotate", False)),
        "misalign_text_image": bool(args.misalign_text_image),
        "reasoning_calib": bool(getattr(args, "reasoning_calib", False)),
    }


def check_args(args):
    """Validate --calib_cache_dir against its calib_cache.json (called once from utils.parser_gen)."""
    cache_dir = args.calib_cache_dir
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"--calib_cache_dir {cache_dir} is not a directory")
    manifest_path = os.path.join(cache_dir, MANIFEST)
    with open(manifest_path) as f:
        expected = json.load(f)["config"]
    got = run_config(args)
    diff = {k: (expected.get(k), got.get(k)) for k in set(expected) | set(got) if expected.get(k) != got.get(k)}
    if diff:
        raise ValueError(
            f"--calib_cache_dir {cache_dir}: the run's calibration settings differ from {MANIFEST} "
            f"(name: (cache, run)): {diff}"
        )
    for name in FILES.values():
        path = os.path.join(cache_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--calib_cache_dir: missing {path}")
    if not args.use_cache or args.act_calib_debug:
        raise ValueError("--calib_cache_dir needs --use_cache and no --act_calib_debug (the files are read only)")
    logging.info("Using released calibration caches from %s", cache_dir)


def released_file(args, kind):
    """Path of a released cache file, or None when --calib_cache_dir is not set (or kind is not released)."""
    cache_dir = getattr(args, "calib_cache_dir", None)
    if not cache_dir or kind is None:
        return None
    path = os.path.join(cache_dir, FILES[kind])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--calib_cache_dir: missing {path}")
    return path
