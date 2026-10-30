import json
import logging
import os
import re

import torch
from tqdm import tqdm
from . import qwen_vlm_utils


def _strip_special_tokens(text: str) -> str:
    text = text.replace("<s>", "").replace("</s>", "").strip()
    text = re.sub(r"<\|[^|]+?\|>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_qwen2_inputs(message, image_processor, device):
    return qwen_vlm_utils.build_qwen2_eval_inputs(message, image_processor, device)


def _is_multi_choice(line) -> bool:
    qtype = str(line.get("question_type", "")).lower()
    if "multi" in qtype and "choice" in qtype:
        return True
    choices = line.get("choices", None)
    return isinstance(choices, (list, tuple)) and len(choices) > 0


def _extract_mcq_letter(text: str):
    txt = _strip_special_tokens(text).upper()
    patterns = [
        r"ANSWER\s*(?:IS|:|：)?\s*\(?([A-Z])\)?\b",
        r"OPTION\s*(?:IS|:|：)?\s*\(?([A-Z])\)?\b",
        r"CHOICE\s*(?:IS|:|：)?\s*\(?([A-Z])\)?\b",
        r"^\s*\(?([A-Z])\)?\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, txt)
        if matches:
            return matches[-1]
    candidates = re.findall(r"(?<![A-Z])([A-Z])(?![A-Z])", txt)
    if candidates:
        return candidates[-1]
    return None


def _extract_free_form_answer(text: str):
    txt = _strip_special_tokens(text)
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", txt, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if lines:
        return re.sub(r"^(answer\s*[:：]\s*)", "", lines[-1], flags=re.IGNORECASE).strip()
    return txt


def _append_eval_instruction(message, is_multi_choice: bool):
    if not isinstance(message, list):
        return message
    message = list(message)
    if is_multi_choice:
        instruction = (
            "Answer the multiple-choice question. "
            "The last line of your response should be exactly: "
            "'Answer: $LETTER' (without quotes)."
        )
    else:
        instruction = "Answer the question directly. Put only the final answer in the last line."
    message.append({"type": "text", "value": instruction})
    return message


@torch.no_grad()
def evaluator(model, testenc, dev, args, tokenizer, image_processor):
    from vlmeval.smp import dump, get_rank_and_world_size, load, tabulate
    import pandas as pd

    _, dataset = testenc
    rank, world_size = get_rank_and_world_size()
    model = model.to(dev)

    if rank == 0:
        logging.info("Start selecting split data!")

    sheet_indices = list(range(rank, len(dataset), world_size))
    data = dataset.data.iloc[sheet_indices]

    out_file = args.save_path + f"/vlm_eval_rank{rank}.pkl"
    data_indices = [i for i in data["index"]]
    res = {}
    if os.path.exists(out_file):
        res.update(load(out_file))

    data = data[~data["index"].isin(res)]
    if world_size > 1:
        torch.distributed.barrier()
    logging.info("finish selecting split data!")

    for i in tqdm(range(len(data))):
        idx = data.iloc[i]["index"]
        is_multi_choice = _is_multi_choice(data.iloc[i])
        message = dataset.build_prompt(data.iloc[i])
        message = _append_eval_instruction(message, is_multi_choice)
        inputs = _build_qwen2_inputs(message, image_processor, dev)

        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            temperature=0.0,
            top_p=None,
            num_beams=1,
            repetition_penalty=1.0,
            max_new_tokens=512,
        )
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_resp = image_processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        final_text = _strip_special_tokens(raw_resp)
        if is_multi_choice:
            pred = _extract_mcq_letter(raw_resp) or _extract_mcq_letter(final_text)
            if pred is None:
                pred = _extract_free_form_answer(final_text)
        else:
            pred = _extract_free_form_answer(final_text)

        if getattr(args, "verbose", False):
            print(
                f"[idx={idx}] raw={raw_resp}\n"
                f"[idx={idx}] final={final_text}\n"
                f"[idx={idx}] pred={pred}",
                flush=True,
            )

        res[idx] = pred
        if (i + 1) % 10 == 0:
            dump(res, out_file)

    dump({k: res[k] for k in data_indices}, out_file)
    if world_size > 1:
        torch.distributed.barrier()

    if rank == 0:
        merged_results = {}
        for r in range(world_size):
            rank_file = args.save_path + f"/vlm_eval_rank{r}.pkl"
            if os.path.exists(rank_file):
                merged_results.update(load(rank_file))

        merged_out_file = args.save_path + "/vlm_eval.pkl"
        dump(merged_results, merged_out_file)

        for r in range(world_size):
            rank_file = args.save_path + f"/vlm_eval_rank{r}.pkl"
            if os.path.exists(rank_file):
                os.remove(rank_file)

        logging.info("Merged and cleaned up all per-GPU outputs.")

    if world_size > 1:
        torch.distributed.barrier()
    rank, _ = get_rank_and_world_size()
    if rank == 0:
        logging.info("Start evaluation!")
        result_file = args.save_path + "/vlm_result.xlsx"
        data_all = load(args.save_path + "/vlm_eval.pkl")
        eval_data = dataset.data
        for x in eval_data["index"]:
            assert x in data_all

        eval_data["prediction"] = [str(data_all[x]) for x in eval_data["index"]]
        if "image" in eval_data:
            eval_data.pop("image")

        dump(eval_data, result_file)
        judge_kwargs = {"nproc": 4, "verbose": bool(getattr(args, "verbose", False)), "retry": 3}
        eval_results = dataset.evaluate(result_file, **judge_kwargs)
        logging.info(eval_results)
        if isinstance(eval_results, dict):
            logging.info("\n" + json.dumps(eval_results, indent=4))
        elif isinstance(eval_results, pd.DataFrame):
            if len(eval_results) < len(eval_results.columns):
                eval_results = eval_results.T
            logging.info("\n" + tabulate(eval_results))
        return eval_results
