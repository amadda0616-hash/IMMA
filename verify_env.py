import torch

# 1. CUDA 및 디바이스 확인
print(f"CUDA Available : {torch.cuda.is_available()}")
print(f"Device Name    : {torch.cuda.get_device_name(0)}")
print(f"CUDA Version   : {torch.version.cuda}")

# 2. Blackwell bfloat16 지원
print(f"BFloat16       : {torch.cuda.is_bf16_supported()}")

# 3. flash-attn
try:
    import flash_attn
    print(f"FlashAttn-2    : {flash_attn.__version__}")
except ImportError:
    print("FlashAttention-2 NOT installed.")

# 4. bitsandbytes
try:
    import bitsandbytes  # noqa: F401
    print("bitsandbytes   : OK")
except ImportError:
    print("bitsandbytes NOT installed.")