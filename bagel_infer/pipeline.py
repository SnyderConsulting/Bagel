"""Inference pipelines for Bagel reference-guided editing."""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache

from .factory import InferenceProcessors


def load_image(path: str) -> Image.Image:
    """Load an image from *path* as RGB."""

    image = Image.open(path)
    return pil_img2rgb(image)


def _unpatchify(latents_patched: torch.Tensor, p: int) -> torch.Tensor:
    """Inverse of patchify for latent tensors.

    Converts [B, C*(p*p), H, W] -> [B, C, H*p, W*p].
    """

    if latents_patched.dim() != 4:
        raise ValueError(
            f"Expected 4D tensor for unpatchify, got shape {tuple(latents_patched.shape)}"
        )

    B, Cpp, H, W = latents_patched.shape
    if (Cpp % (p * p)) != 0:
        raise ValueError(
            f"Channel dim {Cpp} must be divisible by p^2={p * p} for unpatchify"
        )

    C = Cpp // (p * p)
    latents = latents_patched.view(B, C, p, p, H, W)
    latents = latents.permute(0, 1, 4, 2, 5, 3).contiguous()
    latents = latents.view(B, C, H * p, W * p)
    return latents


def _move_to_device(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to(device)
    return inputs


def _clone_context(context: Dict[str, object]) -> Dict[str, object]:
    return {
        "kv_lens": list(context["kv_lens"]),
        "ropes": list(context["ropes"]),
        "past_key_values": deepcopy(context["past_key_values"]),
    }


def _decode_latent(latent: torch.Tensor, image_shape: Tuple[int, int], model) -> Image.Image:
    H, W = image_shape
    h = H // model.latent_downsample
    w = W // model.latent_downsample
    if latent.dim() != 2:
        raise RuntimeError(
            f"Unexpected latent shape {tuple(latent.shape)}; expected [num_tokens, latent_dim]"
        )
    if latent.shape[0] != h * w:
        raise RuntimeError(
            f"Latent tokens {latent.shape[0]} do not match grid {h}x{w}"
        )

    latent_patch = getattr(model, "latent_patch_size", getattr(model.config, "latent_patch_size", 1))
    latent_channel = getattr(model, "latent_channel", getattr(model.config, "latent_channel", None))
    if latent_channel is None:
        latent_channel = getattr(getattr(model, "vae_model", object()), "latent_channels", None)
    if latent_channel is None:
        raise AttributeError("Could not determine latent channels for decoding")

    latent = latent.view(h, w, -1).permute(2, 0, 1).unsqueeze(0)
    vae_channels = getattr(getattr(model, "vae_model", None), "latent_channels", latent_channel)
    if latent.shape[1] != vae_channels:
        latent = _unpatchify(latent, p=latent_patch)

    decoded = model.vae_model.decode(latent)
    if decoded.min() < 0:
        decoded = (decoded.clamp(-1, 1) + 1) / 2
    decoded = decoded.clamp(0, 1)
    image = (decoded[0] * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(image)


@torch.no_grad()
def predict_single_edit(
    model,
    processors: InferenceProcessors,
    ref_path: str,
    input_path: str,
    *,
    device: str = "cuda",
    fp16: bool = True,
    num_timesteps: int = 30,
    cfg_text_scale: float = 1.0,
    cfg_img_scale: float = 1.0,
    cfg_interval: Tuple[float, float] = (0.0, 1.0),
) -> Image.Image:
    """Generate an edited prediction for a single (reference, input) pair."""

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device_obj = torch.device(device)

    model = model.to(device_obj)
    model.eval()
    if not hasattr(model, "vae_model"):
        raise AttributeError("Model is missing attached VAE model; call build_model first")
    model.vae_model = model.vae_model.to(device_obj).eval()

    amp_dtype = torch.bfloat16 if fp16 and device_obj.type == "cuda" else torch.float32

    input_image = load_image(input_path)
    ref_image = load_image(ref_path)

    main_context: Dict[str, object] = {
        "kv_lens": [0],
        "ropes": [0],
        "past_key_values": NaiveCache(model.config.llm_config.num_hidden_layers),
    }

    def _update_context(image):
        nonlocal main_context
        past_key_values = main_context["past_key_values"]
        kv_lens = main_context["kv_lens"]
        ropes = main_context["ropes"]

        generation_input, kv_lens, ropes = model.prepare_vae_images(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=[image],
            transforms=processors.vae_transform,
            new_token_ids=processors.new_token_ids,
        )
        image_shape = tuple(generation_input["padded_images"].shape[-2:])
        generation_input = _move_to_device(generation_input, device_obj)
        with torch.autocast(device_obj.type, enabled=fp16 and device_obj.type == "cuda", dtype=amp_dtype):
            past_key_values = model.forward_cache_update_vae(
                model.vae_model, past_key_values, **generation_input
            )

        generation_input, kv_lens, ropes = model.prepare_vit_images(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=[image],
            transforms=processors.vit_transform,
            new_token_ids=processors.new_token_ids,
        )
        generation_input = _move_to_device(generation_input, device_obj)
        with torch.autocast(device_obj.type, enabled=fp16 and device_obj.type == "cuda", dtype=amp_dtype):
            past_key_values = model.forward_cache_update_vit(past_key_values, **generation_input)

        main_context = {
            "past_key_values": past_key_values,
            "kv_lens": kv_lens,
            "ropes": ropes,
        }
        return image_shape

    target_shape = _update_context(input_image)
    _update_context(ref_image)

    cfg_text_context = _clone_context(main_context)
    cfg_img_context = _clone_context(main_context)

    generation_input = model.prepare_vae_latent(
        curr_kvlens=main_context["kv_lens"],
        curr_rope=main_context["ropes"],
        image_sizes=[target_shape],
        new_token_ids=processors.new_token_ids,
    )
    generation_input = _move_to_device(generation_input, device_obj)

    generation_input_cfg_text = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_text_context["kv_lens"],
        curr_rope=cfg_text_context["ropes"],
        image_sizes=[target_shape],
    )
    generation_input_cfg_text = _move_to_device(generation_input_cfg_text, device_obj)

    generation_input_cfg_img = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_context["kv_lens"],
        curr_rope=cfg_img_context["ropes"],
        image_sizes=[target_shape],
    )
    generation_input_cfg_img = _move_to_device(generation_input_cfg_img, device_obj)

    timestep_shift = getattr(model.config, "timestep_shift", 1.0)

    with torch.autocast(device_obj.type, enabled=fp16 and device_obj.type == "cuda", dtype=amp_dtype):
        latents = model.generate_image(
            past_key_values=main_context["past_key_values"],
            cfg_text_past_key_values=cfg_text_context["past_key_values"],
            cfg_img_past_key_values=cfg_img_context["past_key_values"],
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=generation_input_cfg_text["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text["cfg_packed_key_value_indexes"],
            cfg_img_packed_position_ids=generation_input_cfg_img["cfg_packed_position_ids"],
            cfg_img_packed_query_indexes=generation_input_cfg_img["cfg_packed_query_indexes"],
            cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img["cfg_packed_key_value_indexes"],
        )

    latent = latents[0]
    return _decode_latent(latent, target_shape, model)


def glob_examples(dataset_root: str) -> List[Tuple[str, str, Optional[str]]]:
    """Return (reference, input, optional ground truth) triples from *dataset_root*."""

    root = Path(dataset_root)
    ref_dir = root / "breast"
    in_dir = root / "input"
    gt_dir = root / "output"

    if not ref_dir.is_dir() or not in_dir.is_dir():
        raise FileNotFoundError(f"Expected '{ref_dir}' and '{in_dir}' under {root}")

    ref_ids = {p.stem for p in ref_dir.glob("*.png")}
    inp_ids = {p.stem for p in in_dir.glob("*.png")}
    ids = sorted(ref_ids & inp_ids, key=lambda s: (len(s), s))

    has_gt = gt_dir.is_dir()
    triples: List[Tuple[str, str, Optional[str]]] = []
    if not has_gt:
        print(f"[infer] No 'output/' found under {dataset_root}; running inference-only (no metrics).")
    for sample_id in ids:
        ref_path = str(ref_dir / f"{sample_id}.png")
        in_path = str(in_dir / f"{sample_id}.png")
        gt_candidate = gt_dir / f"{sample_id}.png"
        gt_path: Optional[str] = str(gt_candidate) if has_gt and gt_candidate.exists() else None
        triples.append((ref_path, in_path, gt_path))
    return triples


@torch.no_grad()
def run_batch(
    model,
    processors: InferenceProcessors,
    dataset_root: str,
    save_dir: str,
    *,
    device: str = "cuda",
    fp16: bool = True,
    num_timesteps: int = 30,
    cfg_text_scale: float = 1.0,
    cfg_img_scale: float = 1.0,
    cfg_interval: Tuple[float, float] = (0.0, 1.0),
) -> List[Tuple[str, Optional[str], str]]:
    """Run inference on every pair in *dataset_root* and save predictions."""

    os.makedirs(save_dir, exist_ok=True)
    pred_dir = os.path.join(save_dir, "preds")
    os.makedirs(pred_dir, exist_ok=True)

    triples = glob_examples(dataset_root)
    results: List[Tuple[str, Optional[str], str]] = []
    for ref_path, in_path, gt_path in triples:
        sample_id = Path(in_path).stem
        pred = predict_single_edit(
            model,
            processors,
            ref_path,
            in_path,
            device=device,
            fp16=fp16,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
        )
        out_path = os.path.join(pred_dir, f"{sample_id}.png")
        pred.save(out_path)
        results.append((out_path, gt_path, sample_id))
    return results
