from vle.inference.base import InferenceBackend, InferenceRequest, InferenceResult, OutOfMemory  # noqa: F401


def make_backend(cfg: dict) -> InferenceBackend:
    """Factory: cfg['backend'] in {'hf', 'vllm', 'mock'}; the same request/response API for all."""
    kind = cfg.get("backend", "mock")
    if kind == "hf":
        from vle.inference.hf_backend import HFVisionLanguageBackend
        return HFVisionLanguageBackend(**cfg.get("hf", {}))
    if kind == "vllm":
        from vle.inference.vllm_backend import VLLMOpenAIBackend
        return VLLMOpenAIBackend(**cfg.get("vllm", {}))
    if kind == "mock":
        from vle.inference.mock_backend import MockBackend
        return MockBackend(**cfg.get("mock", {}))
    raise ValueError(f"unknown backend {kind}")
