# from .model import LlavaLlamaForCausalLM

try:
    from .model import LlavaLlamaForCausalLM
except ImportError as e:
    print(f"ERROR: {e}")
    raise