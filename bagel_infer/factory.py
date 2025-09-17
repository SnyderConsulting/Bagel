"""Factory helpers for constructing Bagel inference components."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer


@dataclass
class InferenceProcessors:
    """Container with reusable preprocessing utilities for inference."""

    tokenizer: Qwen2Tokenizer
    new_token_ids: Dict[str, int]
    vae_transform: ImageTransform
    vit_transform: ImageTransform


def _find_checkpoint_files(ckpt_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Return candidate (ema, main) checkpoint files inside *ckpt_dir*."""

    candidates = [
        "ema.safetensors",
        "model_ema.safetensors",
        "model.safetensors",
        "pytorch_model.bin",
        "model.pt",
        "full_state_dict.pt",
    ]
    found = [p for p in Path(ckpt_dir).glob("*") if p.name in candidates]
    ema = next((str(p) for p in found if "ema" in p.name), None)
    main = next((str(p) for p in found if "ema" not in p.name), None)
    return ema, main


def _step_dirs(root: str) -> list[Path]:
    """Return numeric step subdirectories inside *root*, sorted by step."""

    return sorted(
        [p for p in Path(root).iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )


def _latest_checkpoint_dir(root: str) -> str:
    """Return the numerically latest checkpoint directory inside *root*."""

    dirs = _step_dirs(root)
    if not dirs:
        raise FileNotFoundError(f"No step subdirs found in {root}")
    return str(dirs[-1])


def resolve_checkpoint_dir(checkpoint_root: str, step: Optional[int]) -> str:
    """Resolve a checkpoint directory given an optional *step* override."""

    if step is None:
        return _latest_checkpoint_dir(checkpoint_root)
    dirs = _step_dirs(checkpoint_root)
    width = len(dirs[-1].name) if dirs else 7
    return os.path.join(checkpoint_root, f"{step:0{width}d}")


def load_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    """Load model weights from *ckpt* (file or directory).

    Prefers EMA weights when present.
    """

    if os.path.isdir(ckpt):
        ema, main = _find_checkpoint_files(ckpt)
        ckpt_path = ema or main
        if ckpt_path is None:
            raise FileNotFoundError(f"No model weights found under {ckpt}")
    else:
        ckpt_path = ckpt

    ext = os.path.splitext(ckpt_path)[1]
    if ext == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[load_checkpoint] loaded {ckpt_path}")
    if missing:
        print(f"[load_checkpoint] missing keys: {len(missing)} (ok if frozen parts)")
    if unexpected:
        print(f"[load_checkpoint] unexpected keys: {len(unexpected)} (likely optimizer/FSDP)")


def build_processors(
    llm_path: str,
    *,
    vae_kwargs: Optional[Dict[str, int]] = None,
    vit_kwargs: Optional[Dict[str, int]] = None,
) -> InferenceProcessors:
    """Build tokenizer + image transforms used during inference."""

    tokenizer = Qwen2Tokenizer.from_pretrained(llm_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_defaults = dict(image_stride=16, max_image_size=512, min_image_size=256)
    vit_defaults = dict(image_stride=14, max_image_size=350, min_image_size=224)
    if vae_kwargs:
        vae_defaults.update(vae_kwargs)
    if vit_kwargs:
        vit_defaults.update(vit_kwargs)

    vae_transform = ImageTransform(**vae_defaults)
    vit_transform = ImageTransform(**vit_defaults)

    return InferenceProcessors(
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
    )


def _load_llm_config(model_path: str, llm_path: str) -> Qwen2Config:
    cfg_path = Path(model_path) / "llm_config.json"
    if cfg_path.exists():
        return Qwen2Config.from_json_file(str(cfg_path))
    return Qwen2Config.from_pretrained(llm_path)


def _load_vit_config(model_path: str, vit_path: str) -> SiglipVisionConfig:
    cfg_path = Path(model_path) / "vit_config.json"
    if cfg_path.exists():
        return SiglipVisionConfig.from_json_file(str(cfg_path))
    return SiglipVisionConfig.from_pretrained(vit_path)


def _load_bagel_config(model_path: str) -> Optional[BagelConfig]:
    cfg_path = Path(model_path) / "config.json"
    if cfg_path.exists():
        try:
            return BagelConfig.from_json_file(str(cfg_path))
        except json.JSONDecodeError:
            pass
    return None


def build_model(
    *,
    model_path: str,
    llm_path: str,
    vit_path: str,
    vae_path: str,
    device: str = "cuda",
    max_latent_size: int = 64,
    latent_patch_size: int = 2,
    vit_max_num_patch_per_side: int = 30,
) -> Bagel:
    """Instantiate the Bagel model for inference."""

    torch.set_grad_enabled(False)
    device_obj = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")

    llm_config = _load_llm_config(model_path, llm_path)
    vit_config = _load_vit_config(model_path, vit_path)
    vae_model, vae_config = load_ae(vae_path)

    bagel_config = _load_bagel_config(model_path) or BagelConfig()
    bagel_config.visual_gen = True
    bagel_config.visual_und = True
    bagel_config.llm_config = llm_config
    bagel_config.vit_config = vit_config
    bagel_config.vae_config = vae_config
    bagel_config.latent_patch_size = latent_patch_size
    bagel_config.max_latent_size = max_latent_size
    bagel_config.vit_max_num_patch_per_side = vit_max_num_patch_per_side

    language_model = Qwen2ForCausalLM.from_pretrained(llm_path, config=llm_config)
    vit_model = SiglipVisionModel.from_pretrained(vit_path, config=vit_config)
    if hasattr(vit_model, "vision_model") and hasattr(vit_model.vision_model, "embeddings"):
        converter = getattr(vit_model.vision_model.embeddings, "convert_conv2d_to_linear", None)
        if callable(converter):
            converter(vit_config)

    model = Bagel(language_model, vit_model, bagel_config)
    model.eval().to(device_obj)

    vae_model = vae_model.to(device_obj).eval()
    model.vae_model = vae_model
    model.to(device_obj)

    return model
