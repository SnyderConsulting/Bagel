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
from .patchify import unpatchify_v1, unpatchify_v2
from .vae_io import vae_decode


def load_image(path: str) -> Image.Image:
    """Load image as RGB with our repo's helper (handles grayscale etc.)."""
    return pil_img2rgb(Image.open(path))


def _move_to_device(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to(device)
    return inputs


def _clone_context(ctx: Dict[str, object]) -> Dict[str, object]:
    return {
        "kv_lens": list(ctx["kv_lens"]),
        "ropes": list(ctx["ropes"]),
        "past_key_values": deepcopy(ctx["past_key_values"]),
    }


def _decode_latent(
    latent: torch.Tensor,
    image_shape: Tuple[int, int],
    model,
    *,
    latent_patch_size: int,
    vae_channels: int,
    latent_downsample: int,
) -> Image.Image:
    """Decode Bagel latent tokens to an RGB image."""

    H, W = image_shape
    h, w = H // latent_downsample, W // latent_downsample

    if latent.dim() == 2:
        if latent.shape[0] != h * w:
            raise RuntimeError(f"Token count {latent.shape[0]} != grid {h}x{w}")
        latent = latent.view(h, w, -1).permute(2, 0, 1).unsqueeze(0).contiguous()
    elif latent.dim() != 4:
        raise RuntimeError(f"Unexpected latent rank {latent.dim()}")

    channels = latent.shape[1]
    if channels == vae_channels:
        lat4 = latent
    elif channels == vae_channels * latent_patch_size * latent_patch_size:
        lat4 = unpatchify_v1(latent, latent_patch_size)
        if lat4.shape[1] != vae_channels:
            lat4 = unpatchify_v2(latent, latent_patch_size)
        if lat4.shape[1] != vae_channels:
            raise RuntimeError(
                f"Unpatchify failed: got C={lat4.shape[1]}, want {vae_channels}"
            )
    else:
        raise RuntimeError(
            f"Unexpected channels {channels}; expected {vae_channels} or {vae_channels}*p*p"
        )

    img = vae_decode(model.vae_model, lat4)[0]
    img = (img * 255.0).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(img)


def predict_single_edit(
    model,
    processors: InferenceProcessors,
    ref_path: str,
    input_path: str,
    device: str = "cuda",
    fp16: bool = True,
) -> Image.Image:
    """Generate an edited prediction for a single (reference, input) pair using Bagel's generate path."""

    device_str = str(device)
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_obj = torch.device("cpu")
    else:
        device_obj = torch.device(device)

    model = model.to(device_obj).eval()
    if not hasattr(model, "vae_model"):
        raise AttributeError("Model is missing attached VAE model; call build_model first")
    model.vae_model = model.vae_model.to(device_obj).eval()

    amp_enabled = fp16 and device_obj.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    # 1) Load images
    in_img = load_image(input_path)
    ref_img = load_image(ref_path)

    # 2) Build initial cache
    main_ctx: Dict[str, object] = {
        "kv_lens": [0],
        "ropes": [0],
        "past_key_values": NaiveCache(model.config.llm_config.num_hidden_layers),
    }

    def update_ctx(image: Image.Image) -> Tuple[int, int]:
        nonlocal main_ctx
        pkv = main_ctx["past_key_values"]
        kv_lens = main_ctx["kv_lens"]
        ropes = main_ctx["ropes"]

        # Stream VAE tokens
        vae_inp, kv_lens, ropes = model.prepare_vae_images(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=[image],
            transforms=processors.vae_transform,
            new_token_ids=processors.new_token_ids,
        )
        image_shape = tuple(vae_inp["padded_images"].shape[-2:])
        vae_inp = _move_to_device(vae_inp, device_obj)
        with torch.autocast(device_obj.type, enabled=amp_enabled, dtype=amp_dtype):
            pkv = model.forward_cache_update_vae(model.vae_model, pkv, **vae_inp)

        # Stream ViT tokens
        vit_inp, kv_lens, ropes = model.prepare_vit_images(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            images=[image],
            transforms=processors.vit_transform,
            new_token_ids=processors.new_token_ids,
        )
        vit_inp = _move_to_device(vit_inp, device_obj)
        with torch.autocast(device_obj.type, enabled=amp_enabled, dtype=amp_dtype):
            pkv = model.forward_cache_update_vit(pkv, **vit_inp)

        main_ctx = {"past_key_values": pkv, "kv_lens": kv_lens, "ropes": ropes}
        return image_shape

    # Stream input first (target size), then reference
    target_shape = update_ctx(in_img)
    update_ctx(ref_img)

    # 3) CFG contexts
    cfg_text_ctx = _clone_context(main_ctx)
    cfg_img_ctx = _clone_context(main_ctx)

    # 4) Prepare latent tokens to be generated
    gen_inp = model.prepare_vae_latent(
        curr_kvlens=main_ctx["kv_lens"],
        curr_rope=main_ctx["ropes"],
        image_sizes=[target_shape],
        new_token_ids=processors.new_token_ids,
    )
    gen_inp = _move_to_device(gen_inp, device_obj)

    cfg_text_inp = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_text_ctx["kv_lens"],
        curr_rope=cfg_text_ctx["ropes"],
        image_sizes=[target_shape],
    )
    cfg_text_inp = _move_to_device(cfg_text_inp, device_obj)

    cfg_img_inp = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_ctx["kv_lens"],
        curr_rope=cfg_img_ctx["ropes"],
        image_sizes=[target_shape],
    )
    cfg_img_inp = _move_to_device(cfg_img_inp, device_obj)

    # 5) Generate latent tokens
    with torch.autocast(device_obj.type, enabled=amp_enabled, dtype=amp_dtype):
        latents = model.generate_image(
            past_key_values=main_ctx["past_key_values"],
            cfg_text_past_key_values=cfg_text_ctx["past_key_values"],
            cfg_img_past_key_values=cfg_img_ctx["past_key_values"],
            num_timesteps=30,
            cfg_text_scale=1.0,
            cfg_img_scale=1.0,
            cfg_interval=(0.0, 1.0),
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
            timestep_shift=getattr(model.config, "timestep_shift", 1.0),
            **gen_inp,
            cfg_text_packed_position_ids=cfg_text_inp["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=cfg_text_inp["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=cfg_text_inp["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=cfg_text_inp["cfg_packed_key_value_indexes"],
            cfg_img_packed_position_ids=cfg_img_inp["cfg_packed_position_ids"],
            cfg_img_packed_query_indexes=cfg_img_inp["cfg_packed_query_indexes"],
            cfg_img_key_values_lens=cfg_img_inp["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=cfg_img_inp["cfg_packed_key_value_indexes"],
        )

    latent = latents[0]

    # 6) Decode to RGB
    patch = getattr(getattr(model, "config", model), "latent_patch_size", 2)
    vae_channels = getattr(getattr(model, "vae_model", None), "latent_channels", 4)
    downsample = getattr(
        model,
        "latent_downsample",
        getattr(processors.vae_transform, "image_stride", 16),
    )

    img = _decode_latent(
        latent,
        target_shape,
        model,
        latent_patch_size=patch,
        vae_channels=vae_channels,
        latent_downsample=downsample,
    )
    setattr(img, "_debug_tag", "bagel_generate")
    return img


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
        print(
            f"[infer] No 'output/' found under {dataset_root}; running inference-only (no metrics)."
        )
    for sample_id in ids:
        ref_path = str(ref_dir / f"{sample_id}.png")
        in_path = str(in_dir / f"{sample_id}.png")
        gt_candidate = gt_dir / f"{sample_id}.png"
        gt_path: Optional[str] = str(gt_candidate) if has_gt and gt_candidate.exists() else None
        triples.append((ref_path, in_path, gt_path))
    return triples


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

    del num_timesteps, cfg_text_scale, cfg_img_scale, cfg_interval  # unused but kept for API

    os.makedirs(save_dir, exist_ok=True)
    pred_dir = os.path.join(save_dir, "preds")
    os.makedirs(pred_dir, exist_ok=True)

    triples = glob_examples(dataset_root)
    results: List[Tuple[str, Optional[str], str]] = []
    for ref_path, in_path, gt_path in triples:
        sample_id = Path(in_path).stem
        pred_img = predict_single_edit(
            model,
            processors,
            ref_path,
            in_path,
            device=device,
            fp16=fp16,
        )
        tag = getattr(pred_img, "_debug_tag", "bagel_generate")
        out_path = os.path.join(pred_dir, f"{sample_id}__{tag}.png")
        pred_img.save(out_path)
        results.append((out_path, gt_path, sample_id))
    return results
