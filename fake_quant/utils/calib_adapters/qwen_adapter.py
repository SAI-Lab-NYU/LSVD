from .base import (
    ROLE_FINAL_ANSWER,
    ROLE_REASONING,
    ReasoningAdapter,
    ReasoningBatch,
    clean_generation_for_replay,
    find_final_start_prefix,
    first_token_divergence,
    replay_batch_from_prompt_inputs,
    role_mask_from_labels,
    sample_with_answer,
)
from .. import qwen_vlm_utils


class QwenReasoningAdapter(ReasoningAdapter):
    name = "qwen"

    @classmethod
    def supports(cls, args=None, model=None, tokenizer=None, image_processor=None) -> bool:
        return qwen_vlm_utils.resolve_qwen_vlm_family(
            args=args,
            model=model,
            tokenizer=tokenizer,
        ) == "qwen3vl"

    def cache_tag(self) -> str:
        return qwen_vlm_utils.QWEN3_VLMEVALKIT_PROMPT_VERSION

    def _with_eval_instruction(self, sample):
        return qwen_vlm_utils.qwen3_vlmevalkit_aligned_message(sample)

    def build_prompt_inputs(self, sample, device):
        return qwen_vlm_utils.build_qwen_vlm_eval_inputs(
            self._with_eval_instruction(sample),
            self.image_processor,
            device,
            args=self.args,
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def decode_generated(self, generated_ids) -> str:
        return self.image_processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]

    def _train_inputs(self, sample, device):
        sample = self._with_eval_instruction(sample)
        explicit_args = type(
            "ReasoningQwenArgs",
            (),
            {
                "model": getattr(self.args, "model", ""),
                "mask_thinking_tokens": False,
            },
        )()
        return qwen_vlm_utils.build_qwen_vlm_train_inputs(
            sample,
            self.image_processor,
            device,
            args=explicit_args,
            model=self.model,
            tokenizer=self.tokenizer,
            supervision_mode="answer_only",
        )

    def prompt_text_from_sample(self, sample) -> str:
        return super().prompt_text_from_sample(self._with_eval_instruction(sample))

    def _assistant_prefix_start(self, sample, prefix_text, full_inputs, device):
        if prefix_text is None:
            return None
        try:
            prefix_inputs, _, _ = self._train_inputs(sample_with_answer(sample, prefix_text), device)
            return first_token_divergence(full_inputs["input_ids"], prefix_inputs["input_ids"])
        except Exception:
            return None

    def build_train_batch(self, sample, trace, device) -> ReasoningBatch:
        if trace.generated_token_ids is not None:
            prompt_inputs = self.build_prompt_inputs(sample, device)
            batch = replay_batch_from_prompt_inputs(
                prompt_inputs=prompt_inputs,
                generated_token_ids=trace.generated_token_ids,
                tokenizer=self.image_processor.tokenizer,
                final_text=trace.final_text,
                default_role=ROLE_REASONING if trace.reasoning_text else ROLE_FINAL_ANSWER,
            )
            batch.meta.update(trace.to_dict())
            batch.meta["role_mask_source"] = "qwen_adapter_exact_tokens"
            return batch

        replay_text = clean_generation_for_replay(trace.raw_generation)
        inputs, labels, meta = self._train_inputs(
            sample_with_answer(sample, replay_text),
            device,
        )
        default_role = ROLE_REASONING if trace.reasoning_text else ROLE_FINAL_ANSWER
        prefix = find_final_start_prefix(replay_text, trace.final_text)
        final_start = self._assistant_prefix_start(sample, prefix, inputs, device)
        role_mask = role_mask_from_labels(
            labels,
            default_role=default_role,
            prompt_len=meta.get("prompt_token_length"),
            final_start=final_start,
        )
        meta = dict(meta)
        meta.update(trace.to_dict())
        meta["replay_generation"] = replay_text
        meta["role_mask_source"] = "qwen_adapter"
        return ReasoningBatch(inputs=inputs, labels=labels, role_mask=role_mask, meta=meta)
