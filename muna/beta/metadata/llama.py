# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

def _validate_llama_cpp_model(model: "llama_cpp.llama.Llama") -> "llama_cpp.llama.Llama": # type: ignore
    try:
        from llama_cpp import Llama
        if not isinstance(model, Llama):
            raise ValueError(f"Expected `llama_cpp.llama.Llama` model but got `{type(model).__qualname__}`")
        return model
    except ImportError:
        raise ImportError("llama-cpp-python is required to create this metadata but it is not installed.")