# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

def validate_diffusers_pipeline(pipeline: "diffusers.DiffusionPipeline") -> "diffusers.DiffusionPipeline": # type: ignore
    try:
        from diffusers import DiffusionPipeline
        if not isinstance(pipeline, DiffusionPipeline):
            raise ValueError(f"Expected a `diffusers.DiffusionPipeline` but got `{type(pipeline).__qualname__}`")
        return pipeline
    except ImportError:
        raise ImportError("`diffusers` is required to create this metadata but it is not installed.")