#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from huggingface_hub import hf_hub_download
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field

from ..types import Prediction, Signature
from ..version import __version__

def parse_hf_tag(tag: str) -> HFTag | None:
    """
    Parse a Hugging Face predictor tag, like `hf:owner/repo[@revision]`.

    Parameters:
        tag (str): Predictor tag.

    Returns:
        HFTag: Parsed tag, or `None` if the tag is not a valid Hugging Face tag.
    """
    HF_TAG_PREFIX = "hf:"
    if not tag.startswith(HF_TAG_PREFIX):
        return None
    repo_id, _, revision = tag[len(HF_TAG_PREFIX):].partition("@")
    if repo_id.count("/") != 1 or not all(repo_id.split("/")):
        return None
    return HFTag(repo_id=repo_id, revision=revision or None)

def load_hf_manifest(hf_tag: HFTag) -> HFManifest:
    """
    Load a predictor manifest from Hugging Face.
    """
    manifest_path = download_hf_file(
        hf_tag.repo_id,
        "config.json",
        revision=hf_tag.revision
    )
    manifest = HFManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    return manifest

def download_hf_file(
    repo_id: str,
    filename: str,
    *,
    revision: str | None
) -> Path:
    """
    Download a file from Hugging Face.
    """
    file_path = hf_hub_download(
        repo_id,
        filename,
        revision=revision,
        library_name="muna",
        library_version=__version__
    )
    return Path(file_path)

class HFTag(BaseModel, **ConfigDict(frozen=True)):
    """
    Hugging Face predictor tag.

    Members:
        repo_id (str): Hugging Face repository identifier.
        revision (str): Repository revision.
    """
    repo_id: str = Field(description="Hugging Face repository identifier.")
    revision: str | None = Field(default=None, description="Repository revision.")

class HFManifest(Prediction):
    """
    This is just a raw prediction but with an added `signature` from the predictor
    so that beta clients like `muna.beta.openai` and `muna.beta.anthropic` still work.
    """
    signature: Signature