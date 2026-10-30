from .qwen_adapter import QwenReasoningAdapter


_ADAPTERS = [
    QwenReasoningAdapter,
]


def get_reasoning_adapter(args=None, model=None, tokenizer=None, image_processor=None):
    for adapter_cls in _ADAPTERS:
        if adapter_cls.supports(
            args=args,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
        ):
            return adapter_cls(
                args=args,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
            )
    raise NotImplementedError(
        "No reasoning VLM calibration adapter is available for this model. "
        "Add a calib_adapters/* adapter before enabling --reasoning_calib."
    )
