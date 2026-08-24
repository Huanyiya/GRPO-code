"""Compatibility registrations for model configs newer than Transformers."""


def register_qwen3_5_configs() -> bool:
    """Register SGLang's Qwen3.5 configs when Transformers lacks them.

    The registration is process-local, so callers must invoke this in both the
    driver and Ray worker processes before using ``AutoConfig``.
    """
    from transformers import AutoConfig

    def is_registered(model_type: str) -> bool:
        try:
            AutoConfig.for_model(model_type)
        except ValueError:
            return False
        return True

    missing_qwen3_5 = not is_registered("qwen3_5")
    missing_qwen3_5_text = not is_registered("qwen3_5_text")
    if not (missing_qwen3_5 or missing_qwen3_5_text):
        return False

    try:
        from sglang.srt.configs.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
    except ImportError:
        # Keep unrelated models usable when SGLang/Qwen3.5 support is absent.
        return False

    if missing_qwen3_5_text:
        AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig, exist_ok=True)
    if missing_qwen3_5:
        AutoConfig.register("qwen3_5", Qwen3_5Config, exist_ok=True)
    return True
