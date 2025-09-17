"""Inference pipelines for Bagel reference-guided editing."""
from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .factory import InferenceProcessors
from .patchify import unpatchify_v1, unpatchify_v2
from .vae_io import vae_decode, vae_encode


def predict_single_edit(
    model,
    image_processor,
    ref_path: str,
    input_path: str,
    device: str = "cuda",
    fp16: bool = True,
) -> Image.Image:
    """Generate an edited prediction for a single (reference, input) pair."""

    if image_processor is None:
        raise ValueError("image_processor is required for predict_single_edit")

    device_str = str(device)
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device_obj = torch.device(device_str)

    model = model.to(device_obj)
    model.eval()
    if not hasattr(model, "vae_model"):
        raise AttributeError("Model is missing attached VAE model; call build_model first")
    model.vae_model = model.vae_model.to(device_obj).eval()

    ref_img = Image.open(ref_path).convert("RGB")
    in_img = Image.open(input_path).convert("RGB")

    size_cfg = getattr(image_processor, "size", None)
    crop_cfg = getattr(image_processor, "crop_size", None)
    size_kw = None
    if isinstance(size_cfg, dict):
        if "height" in size_cfg and "width" in size_cfg:
            size_kw = {
                "height": int(size_cfg["height"]),
                "width": int(size_cfg["width"]),
            }
        elif "shortest_edge" in size_cfg:
            se = int(size_cfg["shortest_edge"])
            size_kw = {"height": se, "width": se}
    if size_kw is None:
        size_kw = {"height": 980, "width": 980}

    common = dict(
        return_tensors="pt",
        do_resize=True,
        do_center_crop=False,
        size=size_kw,
    )
    proc_ref = image_processor(images=ref_img, **common)
    proc_in = image_processor(images=in_img, **common)
    pv_ref = proc_ref["pixel_values"].to(device_obj)
    pv_in = proc_in["pixel_values"].to(device_obj)

    np_in = np.array(in_img, dtype=np.float32) / 255.0
    in_tensor = torch.from_numpy(np_in).permute(2, 0, 1).unsqueeze(0).to(device_obj)
    in_tensor = in_tensor * 2 - 1
    z_in = vae_encode(model.vae_model, in_tensor)

    args_list = [
        {"pixel_values_ref": pv_ref, "pixel_values_inp": pv_in},
        {"pixel_values": torch.cat([pv_ref, pv_in], dim=0)},
        {"ref": pv_ref, "inp": pv_in},
    ]
    out = None

    autocast_enabled = fp16 and device_obj.type == "cuda"
    autocast_ctx = (
        torch.autocast(device_type="cuda", enabled=True) if autocast_enabled else nullcontext()
    )
    with torch.no_grad():
        with autocast_ctx:
            for args in args_list:
                try:
                    out = model(**args)
                    break
                except TypeError:
                    continue
    if out is None:
        raise RuntimeError(
            "Model.forward signature not recognized; please expose an 'infer_edit' or accept "
            "(pixel_values_ref, pixel_values_inp)."
        )

    pred_lat = None
    if isinstance(out, dict):
        for key in ("pred_latents", "edit_latents", "gen_latents", "latents", "pred"):
            value = out.get(key)
            if torch.is_tensor(value):
                pred_lat = value
                break
    elif torch.is_tensor(out):
        pred_lat = out
    if pred_lat is None or pred_lat.ndim != 4:
        shape = None if pred_lat is None else tuple(pred_lat.shape)
        raise RuntimeError(
            f"Predicted latents missing or wrong rank: got {type(out)} with pred_lat={shape}"
        )

    p = getattr(getattr(model, "config", model), "latent_patch_size", 2)
    vch, vh, vw = z_in.shape[1:]
    candidates: List[Tuple[str, torch.Tensor]] = []
    if pred_lat.shape[1] == vch:
        candidates.append(("abs_direct", pred_lat))
    if pred_lat.shape[1] == vch * p * p:
        candidates.append(("abs_unpatch_v1", unpatchify_v1(pred_lat, p)))
        candidates.append(("abs_unpatch_v2", unpatchify_v2(pred_lat, p)))

    for tag, lat in list(candidates):
        if lat.shape[2:] == (vh, vw):
            candidates.append((f"res_{tag}", z_in + lat))

    for tag, lat in candidates:
        if lat.shape[1:] != (vch, vh, vw):
            continue
        decoded = vae_decode(model.vae_model, lat)[0].detach().cpu()
        decoded = (decoded * 255.0).round().clamp(0, 255).to(torch.uint8)
        img = decoded.permute(1, 2, 0).numpy()
        pil = Image.fromarray(img)
        setattr(pil, "_debug_tag", tag)
        return pil

    raise RuntimeError(
        f"No candidate latent matched VAE shape; pred_lat={tuple(pred_lat.shape)}, "
        f"z_in={tuple(z_in.shape)}, p={p}"
    )


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

    if processors.image_processor is None:
        raise ValueError("processors.image_processor is required for run_batch")

    os.makedirs(save_dir, exist_ok=True)
    pred_dir = os.path.join(save_dir, "preds")
    os.makedirs(pred_dir, exist_ok=True)

    triples = glob_examples(dataset_root)
    results: List[Tuple[str, Optional[str], str]] = []
    for ref_path, in_path, gt_path in triples:
        sample_id = Path(in_path).stem
        pred_img = predict_single_edit(
            model,
            processors.image_processor,
            ref_path,
            in_path,
            device=device,
            fp16=fp16,
        )
        tag = getattr(pred_img, "_debug_tag", "abs_direct")
        out_path = os.path.join(pred_dir, f"{sample_id}__{tag}.png")
        pred_img.save(out_path)
        results.append((out_path, gt_path, sample_id))
    return results
