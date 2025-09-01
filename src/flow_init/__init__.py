__all__ = [
    "config",
    "encoder",
    "flows",
    "losses",
    "infer",
    "train",
    "utils",
    "features",
]

# 优化注意力后端：在 CUDA 可用时尽可能启用更高效的 SDPA 核
try:
    import torch

    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp()
        except Exception:
            pass
        try:
            torch.backends.cuda.enable_mem_efficient_sdp()
        except Exception:
            pass
        try:
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        except Exception:
            pass
except Exception:
    pass
