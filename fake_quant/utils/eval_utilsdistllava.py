from . import (
    utils,
    model_utils,
    quant_utils,
    qwen_vlm_utils,
    vlmeval_cache_utils,
)
import torch
import os
import logging
import json
from tqdm import tqdm
from vlmeval.smp import *
import torch.distributed as dist
import datetime


NATIVE_PROMPT_DATASETS = ('hallusionbench', 'ocrbench', 'pope', 'mmvet')
MANAGED_SCORE_CACHE_DATASETS = ('hallusionbench', 'ocrbench', 'pope', 'mmvet')
HALLUSIONBENCH_RESPONSE_INSTRUCTION = (
    ' Please answer yes or no. '
    'Answer the question using a single word or phrase.'
)
POPE_RESPONSE_INSTRUCTION = ' Please answer yes or no.'


def uses_native_dataset_prompt(dataset_name):
    dataset_lower = dataset_name.lower()
    return any(name in dataset_lower for name in NATIVE_PROMPT_DATASETS)


def uses_managed_score_cache(dataset_name):
    dataset_lower = dataset_name.lower()
    return any(name in dataset_lower for name in MANAGED_SCORE_CACHE_DATASETS)


def evaluation_device_index(rank):
    """Map a distributed rank to its assigned local CUDA device."""

    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    if local_rank < 0:
        raise ValueError(f'LOCAL_RANK must be non-negative; got {local_rank}')
    return local_rank


def apply_dataset_prompt_policy(message, dataset_name):
    """Apply repository-specific prompt policies after dataset prompt creation."""

    dataset_lower = dataset_name.lower()
    if 'hallusionbench' in dataset_lower:
        response_instruction = HALLUSIONBENCH_RESPONSE_INSTRUCTION
    elif dataset_lower == 'pope':
        # Match the constrained Y/N prompt used by VLMEvalKit's Qwen2-VL
        # adapter and avoid ambiguous exact-match outputs.
        response_instruction = POPE_RESPONSE_INSTRUCTION
    else:
        return message
    if not isinstance(message, list):
        raise TypeError(
            f'{dataset_name} prompt must be a list of typed message dictionaries'
        )

    prompt = [dict(item) for item in message]
    for item in reversed(prompt):
        if item.get('type') != 'text':
            continue
        text = str(item.get('value', '')).rstrip()
        instruction = response_instruction.strip()
        if instruction not in text:
            item['value'] = text + response_instruction
        return prompt
    raise ValueError(f'{dataset_name} prompt contains no text message')


def generation_max_new_tokens(dataset_name, args):
    configured = getattr(args, 'eval_max_new_tokens', None)
    if configured is not None:
        if configured <= 0:
            raise ValueError('--eval_max_new_tokens must be positive')
        return configured
    if 'hallusionbench' in dataset_name.lower():
        # Match fake_quant/eval_hallusionbench.py, which was used for the
        # repository's earlier HallusionBench experiments.
        return 512 if getattr(args, 'case_study', False) else 128
    if 'mmvet' in dataset_name.lower():
        return 512
    if 'ocrbench' in dataset_name.lower():
        # Preserve the generation budget used by the repository's original
        # OCRBench evaluator.
        return 256
    if getattr(args, 'case_study', False):
        return 512
    return 16


def vlmeval_judge_kwargs(args):
    kwargs = {
        'nproc': getattr(args, 'vlm_judge_nproc', 4),
        'verbose': True,
        'retry': 3,
    }
    judge_model = getattr(args, 'vlm_judge_model', None)
    if judge_model:
        kwargs['model'] = judge_model
    return kwargs


@torch.no_grad()
def evaluator(model, testenc, dev, args, tokenizer, image_processor):
    from vlmeval.inference import infer_data
    from vlmeval.smp import dump, get_rank_and_world_size, string, load
    from vlmeval.dataset import DATASET_TYPE
    import pandas as pd
    rank, world_size = get_rank_and_world_size()
    if (
        isinstance(testenc, (tuple, list))
        and len(testenc) == 2
        and hasattr(testenc[1], "dataset_name")
    ):
        testenc = testenc[1]
    dataset_name = testenc.dataset_name
    if (
        'mmvet' in dataset_name.lower()
        and not getattr(args, 'defer_vlm_scoring', False)
        and not getattr(args, 'vlm_judge_model', None)
    ):
        raise ValueError(
            'MMVet scoring requires --vlm_judge_model, or use '
            '--defer_vlm_scoring and score the completed workbook on CPU.'
        )
    qwen_family = qwen_vlm_utils.resolve_qwen_vlm_family(
        args=args,
        model=model,
        tokenizer=tokenizer,
    )
    # if world_size > 1:
    #     local_rank = os.environ.get('LOCAL_RANK', 0)
    #     torch.cuda.set_device(int(local_rank))
    #     dist.init_process_group(
    #         backend='nccl',
    #         timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
    #     )
    # torch.cuda.synchronize()
    # Several Gloo ranks may share one GPU; global rank is not a CUDA device
    # index in that configuration.
    local_device = evaluation_device_index(rank)
    model = model.to(local_device)
    logging.info("moving model to dev")
    local_model_fingerprint = getattr(args, "_vlm_model_state_fingerprint", None)
    if local_model_fingerprint is None:
        local_model_fingerprint = vlmeval_cache_utils.model_state_fingerprint(model)
        args._vlm_model_state_fingerprint = local_model_fingerprint
    if world_size > 1:
        model_fingerprints = [None] * world_size
        dist.all_gather_object(model_fingerprints, local_model_fingerprint)
        if len(set(model_fingerprints)) != 1:
            raise RuntimeError(
                "Distributed evaluator ranks do not share the same model-state "
                f"fingerprint: {model_fingerprints}"
            )
    model_fingerprint = local_model_fingerprint
    max_new_tokens = generation_max_new_tokens(dataset_name, args)
    cache_identity = vlmeval_cache_utils.prediction_cache_identity(
        testenc,
        args,
        model_fingerprint,
        max_new_tokens,
    )
    if rank == 0:
        cache_reused, quarantine = vlmeval_cache_utils.prepare_prediction_cache(
            args.save_path,
            cache_identity,
        )
        logging.info(
            "VLM prediction cache identity=%s reused=%s quarantine=%s",
            vlmeval_cache_utils.identity_digest(cache_identity),
            cache_reused,
            quarantine,
        )
    if world_size > 1:
        dist.barrier()


    def build_prompt_llava(line, dataset):
        tgt_path = dataset.dump_image(line)
        question = line['question']
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question 
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        if not args.case_study:
            if len(options):
                question += "\nAnswer with the option's letter from the given choices directly."
            else:
                question += "\nAnswer the question directly."
        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=question))
        return message

    def ensure_image_url(image: str) -> str:
        prefixes = ['http://', 'https://', 'file://', 'data:image;']
        if any(image.startswith(prefix) for prefix in prefixes):
            return image
        if os.path.exists(image):
            return 'file://' + image
        raise ValueError(f'Invalid image: {image}')

    def output_process(answer):
        if "<s>" in answer:
            answer = answer.replace("<s>", "").strip()
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[1].strip()
        elif "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[1].strip()
        elif "assistant\n" in answer:
            answer = answer.split("assistant\n")[1].strip()
        elif "<|end_header_id|>\n\n" in answer:
            answer = answer.split("<|end_header_id|>\n\n")[2].strip()

        if "</s>" in answer:
            answer = answer.split("</s>")[0].strip()
        elif "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0].strip()
        elif "<|eot_id|>" in answer:
            answer = answer.split("<|eot_id|>")[0].strip()
        return answer

    def message_to_prompt(data, image_processor, model, tokenizer, dataset=None):
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from llava.constants import IMAGE_TOKEN_INDEX
        from PIL import Image
        from abc import abstractproperty
        if 'hf_v16' in str(tokenizer):
            content, images = [], []
            for msg in data:
                if msg["type"] == "text":
                    content.append({"type": msg["type"], "text": msg["value"]})
                else:
                    content.append({"type": "image"})
                    images.append(Image.open(msg["value"]).convert("RGB"))
            conversation = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            prompt = image_processor.apply_chat_template(
            conversation, add_generation_prompt=True
            )
            inputs = image_processor(prompt, images, return_tensors="pt").to(
                        "cuda", torch.float16)
            return inputs
        elif qwen_family is not None:
            return qwen_vlm_utils.build_qwen_vlm_eval_inputs(
                data,
                image_processor,
                dev,
                args=args,
                model=model,
                tokenizer=tokenizer,
            )
        elif 'intern' in str(tokenizer):
            from intern_utils import load_image
            message = data
            image_num = len([x for x in message if x['type'] == 'image'])
            max_num = 6
            total_max_num = 64
            max_num = max(1, min(max_num, total_max_num // image_num))
            device = utils.get_dev()
            prompt = reorganize_prompt(message, image_num, dataset) # we do not need dataset here?
            

            if image_num > 1:
                image_path = [x['value'] for x in message if x['type'] == 'image']
                num_patches_list, pixel_values_list = [], []
                for image_idx, file_name in enumerate(image_path):
                    upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU'], dataset)
                    curr_pixel_values = load_image(
                        file_name, max_num=max_num, upscale=upscale_flag).to(device).to(torch.bfloat16)
                    num_patches_list.append(curr_pixel_values.size(0))
                    pixel_values_list.append(curr_pixel_values)
                pixel_values = torch.cat(pixel_values_list, dim=0)
            elif image_num == 1:
                image_path = [x['value'] for x in message if x['type'] == 'image'][0]
                upscale_flag = dataset is not None and listinstr(['MMMU'], dataset)
                pixel_values = load_image(
                    image_path, max_num=max_num, upscale=upscale_flag).to(device).to(torch.bfloat16)
                num_patches_list = [pixel_values.size(0)]
            else:
                pixel_values = None
                num_patches_list = []
            kwargs_default = dict(
                do_sample = False,
                max_new_tokens = 512,
                temperature = 0.6,
                top_p = 0.95
            )
            return {'pixel_values': pixel_values, 'num_patches_list': num_patches_list, 'question': prompt, 'generation_config': kwargs_default}
        system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )

        def concat_tilist(message):
            text, images = "", []
            for item in message:
                if item["type"] == "text":
                    text += item["value"]
                elif item["type"] == "image":
                    text += " <image> "
                    images.append(item["value"])
            return text, images

        content, images = concat_tilist(data)
        images = [Image.open(s).convert("RGB") for s in images]
        image_sizes = [img.size for img in images]
        args = abstractproperty()
        args.image_aspect_ratio = "pad"
        image_tensor = process_images(images, image_processor, args).to(
            "cuda", dtype=torch.float16
        )

        prompt = system_prompt + "USER: " + content + " ASSISTANT: "
        # ADD conv templets
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stopping_criteria = KeywordsStoppingCriteria(
            ["</s>"], tokenizer, input_ids
        )
        return input_ids, (image_tensor, image_sizes), stopping_criteria


    def reorganize_prompt(message, image_num, dataset=None):
        if dataset is not None and listinstr(['MUIRBench'], dataset):
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            images_to_remove = ' '.join(['<image>'] * image_num)
            prompt = prompt.replace(images_to_remove, '')
            for i in range(image_num):
                prompt = prompt.replace('<image>', f'<Image-{i + 1}>', 1)
            prompt = ''.join([f'Image-{i + 1}: <image>\n' for i in range(image_num)]) + prompt
        elif dataset is not None and listinstr(["bmmr"], dataset.lower()):
            if image_num == 1:
                prompt = "\n".join([x["value"] for x in message if x["type"] == "text"])
            else:
                prompt, image_idx = "", 1
                for x in message:
                    if x["type"] == "text":
                        prompt += x["value"]
                    elif x["type"] == "image":
                        image_idx += 1
        elif image_num == 1:
            prompt = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        else:
            prompt, image_idx = '', 1
            for x in message:
                if x['type'] == 'text':
                    prompt += x['value']
                elif x['type'] == 'image':
                    prompt += f'<Image-{image_idx}>'
                    image_idx += 1
            prompt = ''.join([f'Image-{i + 1}: <image>\n' for i in range(image_num)]) + prompt
            images_to_remove = ''.join([f'<Image-{i + 1}>' for i in range(image_num)])
            prompt = prompt.replace(images_to_remove, '')
        return prompt


    def infer_data_llava(lm, args, verbose=False):

        dataset = testenc
        dataset_name = dataset.dataset_name
        rank, world_size = get_rank_and_world_size()
        if rank == 0:
            logging.info("Start selecting split data!")
        # Each rank writes to a unique output file
        sheet_indices = list(range(rank, len(dataset), world_size))
        data = dataset.data.iloc[sheet_indices]
        if args.case_study:
            case_file = args.save_path + '/case_study.jsonl'
            os.makedirs(os.path.dirname(case_file), exist_ok=True)
            case_anw = open(case_file, "w")
            res = {}
            
        else:
            out_file = args.save_path + f'/vlm_eval_rank{rank}.pkl'
            data_indices = [i for i in data['index']]
            res = {}

            if os.path.exists(out_file):
                res.update(
                    vlmeval_cache_utils.safe_load_rank_cache(out_file, load)
                )

            # Check if all results are already computed
            all_finished = all(idx in res for idx in data['index'])

        data = data[~data['index'].isin(res)]
        if world_size > 1:
            dist.barrier()
        
        logging.info("finish selecting split data!")
        for i in tqdm(range(len(data))):
            if args.case_study:
                if i not in [40, 141, 1384]:#[314, 1994, 200, 1733, 1809, 1509, 1684, 904, 1912, 1465]:
                    continue
            idx = data.iloc[i]['index']
            if (
                qwen_family is not None
                or 'math' in dataset_name.lower()
                or uses_native_dataset_prompt(dataset_name)
            ):
                message = dataset.build_prompt(data.iloc[i])
            else:
                message = build_prompt_llava(data.iloc[i], dataset)
            message = apply_dataset_prompt_policy(message, dataset_name)
            if 'hf_v16' in str(tokenizer):
                inputs = message_to_prompt(
                message, image_processor, model, tokenizer
                )
                output = model.generate(
                    **inputs,
                    do_sample=False, 
                    temperature=0,
                    max_new_tokens=max_new_tokens,
                    top_p=None,
                    num_beams=1
                )
                response = image_processor.decode(output[0], skip_special_tokens=True)
                response = output_process(response)
            elif qwen_family is not None:
                inputs = message_to_prompt(message, image_processor, model, tokenizer)
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs, 
                        top_p=0.001,
                        top_k=1,
                        temperature=0.01,
                        repetition_penalty=1.0,
                        max_new_tokens=max_new_tokens)
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                    ]
                torch.cuda.empty_cache()
                decoder = image_processor.tokenizer if hasattr(image_processor, "tokenizer") else image_processor
                out = decoder.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                response = out[0]
            elif 'intern' in str(tokenizer):
                inputs = message_to_prompt(message, image_processor, model, tokenizer, dataset_name)
                with torch.inference_mode():
                    response = model.chat(
                        image_processor,
                        **inputs,
                    )
            else:
                inputs_id, image, stopping_criteria = message_to_prompt(
                    message, image_processor, model, tokenizer
                ) 
                image, image_sizes = image
                with torch.inference_mode():
                    if not args.dosample:
                        response = model.generate(
                            inputs_id, images=image,
                            image_sizes = image_sizes,
                            do_sample=False,
                            max_new_tokens=max_new_tokens,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                        )
                    else:
                        response = model.generate(
                            inputs_id, images=image,
                            image_sizes = image_sizes,
                            do_sample=True, temperature=0.1,
                            max_new_tokens=max_new_tokens,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                        )
                torch.cuda.empty_cache()
                response = tokenizer.batch_decode(response, skip_special_tokens=True)[0].strip()
            if args.verbose:
                print(response, flush=True)
            if args.case_study:
                import json
                case_anw.write(json.dumps(
                    {"case_id": i,
                    "input":message,
                    "output":response,}) + "\n")
            else:
                res[idx] = response
                if (i + 1) % 10 == 0:
                    dump(res, out_file)

        if args.case_study:
            logging.info("finish case study!")
            return
        dump({k: res[k] for k in data_indices}, out_file)
        if world_size > 1:
            dist.barrier()
        # Merge results on rank 0 after inference
        if rank == 0:
            merged_results = {}
            for r in range(world_size):
                rank_file = args.save_path + f'/vlm_eval_rank{r}.pkl'
                if os.path.exists(rank_file):
                    merged_results.update(load(rank_file))

            # Dump merged results
            merged_out_file = args.save_path + '/vlm_eval.pkl'
            dump(merged_results, merged_out_file)

            # Clean up per-GPU files
            for r in range(world_size):
                rank_file = args.save_path + f'/vlm_eval_rank{r}.pkl'
                if os.path.exists(rank_file):
                    os.remove(rank_file)

            logging.info("Merged and cleaned up all per-GPU outputs.")

        return
    if world_size > 1:
        dist.barrier()
    # Run inference
    infer_data_llava(model, args, True)
    if args.case_study:
        return # early exit without evaluation
    # Rank 0 handles evaluation
    rank, _ = get_rank_and_world_size()
    if rank == 0:
        logging.info("Start evaluation!")
        result_file = args.save_path + '/vlm_result.xlsx'
        data_all = load(args.save_path + '/vlm_eval.pkl')
        dataset = testenc.data

        # Ensure all indices are covered
        for x in dataset['index']:
            assert x in data_all

        dataset['prediction'] = [str(data_all[x]) for x in dataset['index']]
        if 'image' in dataset:
            dataset.pop('image')

        dump(dataset, result_file)
        vlmeval_cache_utils.mark_prediction_cache_complete(
            args.save_path,
            cache_identity,
            result_file,
        )
        if getattr(args, 'defer_vlm_scoring', False):
            logging.info(
                "Deferred CPU scoring for %s; complete workbook written to %s",
                args.eval_dataset,
                result_file,
            )
            return None

        judge_kwargs = vlmeval_judge_kwargs(args)
        score_identity = score_manifest = expected_score_file = None
        if uses_managed_score_cache(dataset_name):
            judge_model = getattr(args, 'vlm_judge_model', None)
            score_identity, score_manifest, quarantine = (
                vlmeval_cache_utils.prepare_scoring_cache(
                    dataset_name,
                    result_file,
                    judge_model,
                )
            )
            expected_score_file = (
                vlmeval_cache_utils.expected_capability_score_file(
                    dataset_name,
                    result_file,
                    judge_model,
                )
            )
            logging.info(
                "Prepared scoring cache identity=%s quarantine=%s",
                vlmeval_cache_utils.identity_digest(score_identity),
                quarantine,
            )
        eval_results = testenc.evaluate(result_file, **judge_kwargs)
        if expected_score_file is not None:
            if not expected_score_file.is_file() or expected_score_file.stat().st_size == 0:
                raise RuntimeError(
                    "VLMEvalKit did not create the expected score artifact: "
                    f"{expected_score_file}"
                )
            vlmeval_cache_utils.mark_scoring_cache_complete(
                score_manifest,
                score_identity,
                expected_score_file,
            )
        import json
        logging.info(eval_results)
        if isinstance(eval_results, dict):
            logging.info('\n' + json.dumps(eval_results, indent=4))
        elif isinstance(eval_results, pd.DataFrame):
            if len(eval_results) < len(eval_results.columns):
                eval_results = eval_results.T
            logging.info('\n' + tabulate(eval_results))
        return eval_results
