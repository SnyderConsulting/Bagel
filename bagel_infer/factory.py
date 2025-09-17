"""Factory helpers for constructing Bagel inference components."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoConfig, AutoImageProcessor
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
    image_processor: Optional[AutoImageProcessor] = None


def _find_checkpoint_files(
    ckpt_dir: str, prefer_ema: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    """Return preferred (primary, secondary) checkpoint files inside *ckpt_dir*."""

    path = Path(ckpt_dir)
    ema = next((str(p) for p in path.glob("*ema*.safetensors")), None)
    full = next((str(p) for p in path.glob("model.safetensors")), None)

    if prefer_ema:
        primary, secondary = ema, full
    else:
        primary, secondary = full, ema

    if primary or secondary:
        return primary, secondary

    candidates = [
        "ema.safetensors",
        "model_ema.safetensors",
        "model.safetensors",
        "pytorch_model.bin",
        "model.pt",
        "full_state_dict.pt",
    ]
    found = [p for p in path.glob("*") if p.name in candidates]
    ema = next((str(p) for p in found if "ema" in p.name), None)
    main = next((str(p) for p in found if "ema" not in p.name), None)
    return (ema, main) if prefer_ema else (main, ema)


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


def load_checkpoint(
    model: torch.nn.Module, ckpt: str, prefer_ema: bool = False
) -> None:
    """Load model weights from *ckpt* (file or directory).

    Prefers consolidated weights (model.safetensors) unless *prefer_ema* is True
    or the directory only contains EMA checkpoints. Also validates critical
    geometry before loading.
    """

    def _infer_llm_h_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
        for key, value in state_dict.items():
            if (
                "language_model" in key
                and "self_attn.q_proj.weight" in key
                and hasattr(value, "ndim")
                and value.ndim == 2
            ):
                return int(value.shape[0])
        return None

    if os.path.isdir(ckpt):
        primary, secondary = _find_checkpoint_files(ckpt, prefer_ema=prefer_ema)
        ckpt_path = primary or secondary
        if ckpt_path is None:
            raise FileNotFoundError(f"No model weights found under {ckpt}")
        meta_path = os.path.join(ckpt, "checkpoint_meta.json")
    else:
        ckpt_path = ckpt
        meta_path = os.path.join(os.path.dirname(ckpt_path), "checkpoint_meta.json")

    ext = os.path.splitext(ckpt_path)[1]
    if ext == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    ckpt_hidden: Optional[int] = None
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            ckpt_hidden = int(meta.get("llm_hidden_size")) if meta.get("llm_hidden_size") else None
        except Exception:
            ckpt_hidden = None
    if ckpt_hidden is None:
        ckpt_hidden = _infer_llm_h_from_state_dict(state_dict)

    built_hidden = model.language_model.config.hidden_size
    if ckpt_hidden and built_hidden != ckpt_hidden:
        raise RuntimeError(
            "Checkpoint/graph mismatch detected: "
            f"checkpoint hidden_size={ckpt_hidden} vs built model hidden_size={built_hidden}. "
            "Set --llm_path to the config used during training or rebuild the model with that config."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[load_checkpoint] loaded {ckpt_path}")
    if missing:
        print(f"[load_checkpoint] missing keys: {len(missing)} (frozen or absent)")
    if unexpected:
        print(f"[load_checkpoint] unexpected keys: {len(unexpected)} (naming/ema extras)")


def build_processors(
    llm_path: str,
    vit_path: str,
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

    # If a preprocessor_config.json exists locally, this works fully offline.
    image_processor = AutoImageProcessor.from_pretrained(
        vit_path, local_files_only=True, trust_remote_code=False
    )

    return InferenceProcessors(
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        image_processor=image_processor,
    )


def _load_llm_config(llm_path: str) -> Qwen2Config:
    auto_cfg = AutoConfig.from_pretrained(
        llm_path, local_files_only=True, trust_remote_code=False
    )
    if isinstance(auto_cfg, Qwen2Config):
        return auto_cfg
    return Qwen2Config.from_dict(auto_cfg.to_dict())


def _extract_vit_geometry(cfg_dict: dict) -> dict:
    """
    Return a flat dict with at least: hidden_size, num_attention_heads, intermediate_size.
    Handles both HF SiglipConfig (with nested 'vision_config') and plain vision configs.
    """

    v = cfg_dict.get("vision_config") or cfg_dict
    out = {
        "hidden_size": v.get("hidden_size"),
        "num_attention_heads": v.get("num_attention_heads"),
        "intermediate_size": v.get("intermediate_size") or v.get("mlp_dim"),
    }
    # Optional extras that some configs carry
    for k in ("patch_size", "image_size", "rope", "qkv_bias", "layer_norm_eps", "num_hidden_layers"):
        if k in v:
            out[k] = v[k]
    return out


def _load_vit_config(vit_path: str, model_path: str) -> SiglipVisionConfig:
    """Load the SigLIP vision config used during training.

    Prefers ``vit_config.json`` saved alongside the Bagel checkpoint to ensure
    geometry matches the training graph. Falls back to reading the config from
    ``vit_path`` (which must then be compatible with the stored weights).
    """

    vit_cfg_path = Path(model_path) / "vit_config.json"
    if vit_cfg_path.is_file():
        with open(vit_cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        # Extract geometry robustly
        geom = _extract_vit_geometry(cfg_dict)
        vit_cfg = SiglipVisionConfig.from_dict(geom)
        # Force critical dims in case from_dict applied defaults
        if geom.get("hidden_size") is not None:
            vit_cfg.hidden_size = geom["hidden_size"]
        if geom.get("num_attention_heads") is not None:
            vit_cfg.num_attention_heads = geom["num_attention_heads"]
        if geom.get("intermediate_size") is not None:
            vit_cfg.intermediate_size = geom["intermediate_size"]
        # Optional flags set during training
        if "rope" in geom:
            vit_cfg.rope = geom["rope"]
        vit_cfg._name_or_path = vit_path
        return vit_cfg

    auto_cfg = AutoConfig.from_pretrained(
        vit_path, local_files_only=True, trust_remote_code=True
    )
    if isinstance(auto_cfg, SiglipVisionConfig):
        vit_cfg = auto_cfg
    else:
        vit_cfg = SiglipVisionConfig.from_dict(auto_cfg.to_dict())
    if not getattr(vit_cfg, "_name_or_path", None):
        vit_cfg._name_or_path = vit_path
    return vit_cfg


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

    llm_config = _load_llm_config(llm_path)
    llm_config.qk_norm = True
    if hasattr(llm_config, "k_norm"):
        llm_config.k_norm = True
    # Ensure the decoder layers match training (MoT) so *_moe_gen weights load.
    setattr(llm_config, "layer_module", "Qwen2MoTDecoderLayer")
    if hasattr(llm_config, "tie_word_embeddings"):
        llm_config.tie_word_embeddings = False

    vit_config = _load_vit_config(vit_path, model_path)
    if hasattr(vit_config, "rope"):
        vit_config.rope = False
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
    moe_keys_in_model = sum("moe_gen" in k for k in language_model.state_dict().keys())
    assert (
        moe_keys_in_model > 0
    ), "LLM built without MoT; expected *_moe_gen params for checkpoint compatibility."
    vit_model = SiglipVisionModel.from_pretrained(vit_path, config=vit_config)
    if hasattr(vit_model, "vision_model") and hasattr(vit_model.vision_model, "embeddings"):
        converter = getattr(vit_model.vision_model.embeddings, "convert_conv2d_to_linear", None)
        if callable(converter):
            converter(vit_config)

    model = Bagel(language_model, vit_model, bagel_config)
    built_hidden = model.language_model.config.hidden_size
    if built_hidden != llm_config.hidden_size:
        raise RuntimeError(
            "Mismatch between instantiated LLM hidden size and configuration from --llm_path: "
            f"model={built_hidden} vs cfg={llm_config.hidden_size}"
        )

    def _get_hidden_size_from_cfg(cfg):
        hs = getattr(cfg, "hidden_size", None)
        if hs is not None:
            return hs
        vc = getattr(cfg, "vision_config", None)
        return getattr(vc, "hidden_size", None)

    vit_hidden = _get_hidden_size_from_cfg(vit_config)
    try:
        built_vit_hidden = model.vision_model.config.hidden_size  # type: ignore[attr-defined]
    except AttributeError:
        built_vit_hidden = getattr(getattr(model, "vit_model", None), "config", None)
        built_vit_hidden = getattr(built_vit_hidden, "hidden_size", None)
    if vit_hidden is not None and built_vit_hidden is not None and built_vit_hidden != vit_hidden:
        raise RuntimeError(
            "Mismatch between instantiated ViT hidden size and configuration from vit_config: "
            f"model={built_vit_hidden} vs cfg={vit_hidden}"
        )
    model.eval().to(device_obj)

    vae_model = vae_model.to(device_obj).eval()
    model.vae_model = vae_model
    model.to(device_obj)

    return model
