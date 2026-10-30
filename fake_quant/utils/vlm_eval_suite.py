"""Sequential multi-dataset evaluation for one prepared VLM instance."""

from __future__ import annotations

import os
import logging
from typing import Any

from . import data_utils, utils


def output_tag(dataset: str) -> str:
    dataset_lower = dataset.lower()
    if "scienceqa" in dataset_lower:
        return "sqa"
    if "seedbench" in dataset_lower:
        return "seed"
    if "mmbench" in dataset_lower:
        return "mmbench"
    if "vizwiz" in dataset_lower:
        return "vizwiz"
    if "ocrbench" in dataset_lower:
        return "ocrbench"
    if "hallusionbench" in dataset_lower:
        return "hallusion"
    if dataset_lower == "pope":
        return "pope"
    if "mmvet" in dataset_lower:
        return "mmvet"
    return dataset_lower


def evaluate_one_dataset(
    model: Any,
    args: Any,
    tokenizer: Any,
    image_processor: Any,
    *,
    llava_next: bool,
) -> Any:
    testloader = data_utils.get_loaders(
        args.eval_dataset,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        hf_token=args.hf_token,
        eval_mode=True,
    )
    dataset_lower = args.eval_dataset.lower()
    if (
        "scienceqa" in dataset_lower
        or "seedbench" in dataset_lower
        or "mmbench" in dataset_lower
        or any(
            dataset in dataset_lower
            for dataset in ("hallusionbench", "ocrbench", "pope", "mmvet")
        )
    ):
        from . import eval_utilsdistllava as evaluator_module
    elif "vizwiz" in dataset_lower:
        # LSVD release: VizWiz evaluators (utils/eval_llavanext_vizwiz.py,
        # utils/eval_utilsdistvizwiz.py) are not included.
        raise NotImplementedError("VizWiz evaluation is not included in the LSVD release.")
    else:
        # LSVD release: the text-only LM perplexity evaluator
        # (utils/eval_utilstransformersupgrade.py) is not included; it only
        # supported OPT/Llama checkpoints.
        raise NotImplementedError(
            f"Evaluation dataset {args.eval_dataset!r} is not supported in the LSVD release."
        )

    if evaluator_module.__name__.endswith("eval_utilstransformersupgrade"):
        score = evaluator_module.evaluator(
            model,
            testloader,
            utils.get_dev(),
            args,
        )
    else:
        score = evaluator_module.evaluator(
            model,
            testloader,
            utils.get_dev(),
            args,
            tokenizer,
            image_processor,
        )
    del testloader
    utils.cleanup_memory()
    return score


def evaluate_datasets(
    model: Any,
    args: Any,
    tokenizer: Any,
    image_processor: Any,
    *,
    llava_next: bool,
) -> dict[str, Any]:
    """Evaluate an ordered suite without rebuilding or serializing the model."""

    original_eval_dataset = args.eval_dataset
    original_save_path = args.save_path
    eval_datasets = args.eval_datasets or [args.eval_dataset]
    if len(set(eval_datasets)) != len(eval_datasets):
        raise ValueError(f"Duplicate datasets in --eval_datasets: {eval_datasets}")
    base_tag = output_tag(original_eval_dataset)
    base_marker = f"{os.sep}{base_tag}{os.sep}"
    if len(eval_datasets) > 1 and base_marker not in original_save_path:
        raise ValueError(
            "Sequential multi-dataset evaluation needs the base output path to "
            f"contain {base_marker!r}; got {original_save_path!r}."
        )

    scores: dict[str, Any] = {}
    try:
        for eval_dataset in eval_datasets:
            args.eval_dataset = eval_dataset
            if len(eval_datasets) > 1:
                eval_tag = output_tag(eval_dataset)
                args.save_path = original_save_path.replace(
                    base_marker,
                    f"{os.sep}{eval_tag}{os.sep}",
                    1,
                )
                os.makedirs(args.save_path, exist_ok=True)
            logging.info(
                "Starting in-memory evaluation: dataset=%s output=%s",
                args.eval_dataset,
                args.save_path,
            )
            scores[eval_dataset] = evaluate_one_dataset(
                model,
                args,
                tokenizer,
                image_processor,
                llava_next=llava_next,
            )
    finally:
        args.eval_dataset = original_eval_dataset
        args.save_path = original_save_path
    return scores
