# Third-party dependencies

The LSVD code imports three external packages that are installed from source at the exact
commits used for the paper. `scripts/setup/install_third_party.sh` clones them into
`third_party/src/` (or `$LSVD_THIRD_PARTY_DIR`), applies the local patches below and installs them.

| Package | Upstream | Commit | Local changes | Install |
|---|---|---|---|---|
| `llava` | https://github.com/haotian-liu/LLaVA | `c121f0432da27facab705978f83c4ada465e46fd` | `llava_cache_position.patch` | `pip install --no-deps -e` |
| `vlmeval` | https://github.com/open-compass/VLMEvalKit | `00804217f868058f871f5ff252a7b9623c3475d9` | `vlmevalkit_local_changes.patch` | `pip install -e` |
| `fast_hadamard_transform` | https://github.com/Dao-AILab/fast-hadamard-transform | `e7706faf8d1c3b9f241e36860640ad1dac644ede` | none | `pip install --no-build-isolation` |

## LLaVA

Used to load the LLaVA-v1.5 checkpoints (`liuhaotian/llava-v1.5-{7b,13b}`); LLaVA-Next models
are loaded through `transformers` (`llava-hf/llava-v1.6-vicuna-{7b,13b}-hf`).

- `llava_cache_position.patch` adds a `cache_position=None` keyword to
  `LlavaLlamaForCausalLM.forward`, which `transformers==4.52.4` passes during `generate()`.
- Install with `--no-deps`: LLaVA's `pyproject.toml` pins `torch==2.1.2` and
  `transformers==4.37.2`, which must not replace the versions in `requirements.txt`.
- The `myllava/` copy that exists in the development repository is not needed; the paper runs
  imported the upstream package above.

## VLMEvalKit

Provides the ScienceQA-IMG, SEED-Bench-IMG and MMBench (EN v1.1) datasets and scoring.
`vlmevalkit_local_changes.patch` contains the two local edits present in the paper environment:

1. `run.py`: `os.environ['SPLIT_THINK'] = 'True'`. Only VLMEvalKit's own `run.py` CLI sets it;
   the LSVD evaluators call VLMEvalKit as a library and do not import `run.py`.
2. `vlmeval/dataset/image_mcq.py`: the expected md5 of `MMBench_TEST_EN_V11` is changed from
   `26f0f15381a21720255091d3e0316ce6` to `aab598a927c183aa6657de9097e2be09`, the md5 of the
   `MMBench_TEST_EN_V11.tsv` file used for the paper. Without this edit VLMEvalKit treats that
   local TSV as stale and downloads the current upstream file instead.

VLMEvalKit reads its data directory from `$LMUData` (default `~/LMUData`). It also loads a `.env`
file from its repository root at import time and lets it override the environment; if you use
such a file, set `LMUData` there, or set `WSVD_LMUDATA_ROOT` to the same directory (the LSVD data
loaders copy it back into `LMUData` after importing VLMEvalKit).

md5 of the evaluation TSVs used for the paper:

```
e42e9e00f9c59a80d8a5db35bc32b71f  ScienceQA_TEST.tsv
68017231464752261a2526d6ca3a10c0  SEEDBench_IMG.tsv
aab598a927c183aa6657de9097e2be09  MMBench_TEST_EN_V11.tsv
```

## fast-hadamard-transform

CUDA extension for the online Hadamard transforms (QuaRot-style rotation). It must be compiled
against the installed PyTorch, hence `--no-build-isolation`.

## Triton kernels

The fused Triton kernels used for the efficiency results are not part of the accuracy code in
`fake_quant/`; see `kernels/README.md`.
