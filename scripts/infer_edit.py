"""Run Bagel reference-guided inference over the breast-edit dataset."""
from __future__ import annotations

import argparse
import os

from bagel_infer.factory import (
    build_model,
    build_processors,
    load_checkpoint,
    resolve_checkpoint_dir,
)
from bagel_infer.pipeline import run_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bagel breast-edit inference")
    parser.add_argument("--checkpoint_root", required=True, help="Directory with step checkpoints")
    parser.add_argument("--step", type=int, default=None, help="Optional checkpoint step; defaults to latest")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned Bagel model directory")
    parser.add_argument("--llm_path", required=True, help="Path to Qwen2 checkpoint")
    parser.add_argument("--vit_path", required=True, help="Path to SigLIP vision checkpoint")
    parser.add_argument("--vae_path", required=True, help="Path to VAE weights (ae.safetensors)")
    parser.add_argument("--dataset_root", required=True, help="Root of breast-edit dataset with breast/input/output")
    parser.add_argument("--save_dir", required=True, help="Directory to store predictions")
    parser.add_argument("--ref_path", help="Optional single reference image path (bypasses dataset mode)")
    parser.add_argument("--input_path", help="Optional single input image path (bypasses dataset mode)")
    parser.add_argument("--out_name", default="pred.png", help="Output filename for single-pair mode")
    parser.add_argument("--device", default="cuda", help="Device to run inference on")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Prefer EMA checkpoint weights instead of model.safetensors",
    )
    parser.add_argument("--fp16", action="store_true", help="Enable FP16/bfloat16 autocast on CUDA")
    parser.add_argument("--num_timesteps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--cfg_text_scale", type=float, default=1.0, help="Text guidance scale (keep 1.0 for no CFG)")
    parser.add_argument("--cfg_img_scale", type=float, default=1.0, help="Image guidance scale (keep 1.0 for no CFG)")
    parser.add_argument(
        "--cfg_interval",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        metavar=("START", "END"),
        help="Interval (fraction of steps) to apply CFG",
    )
    parser.add_argument("--vae_max_image_size", type=int, default=512)
    parser.add_argument("--vae_min_image_size", type=int, default=256)
    parser.add_argument("--vae_image_stride", type=int, default=16)
    parser.add_argument("--vit_max_image_size", type=int, default=350)
    parser.add_argument("--vit_min_image_size", type=int, default=224)
    parser.add_argument("--vit_image_stride", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_dir = resolve_checkpoint_dir(args.checkpoint_root, args.step)
    print(f"[infer] using checkpoint dir: {ckpt_dir}")

    processors = build_processors(
        args.llm_path,
        args.vit_path,
        vae_kwargs=dict(
            max_image_size=args.vae_max_image_size,
            min_image_size=args.vae_min_image_size,
            image_stride=args.vae_image_stride,
        ),
        vit_kwargs=dict(
            max_image_size=args.vit_max_image_size,
            min_image_size=args.vit_min_image_size,
            image_stride=args.vit_image_stride,
        ),
    )

    model = build_model(
        model_path=args.model_path,
        llm_path=args.llm_path,
        vit_path=args.vit_path,
        vae_path=args.vae_path,
        device=args.device,
        max_latent_size=64,
        latent_patch_size=2,
        vit_max_num_patch_per_side=30,
    )
    print(
        f"[infer] built LLM hidden={model.language_model.config.hidden_size} (from --llm_path)"
    )
    load_checkpoint(model, ckpt_dir, prefer_ema=args.use_ema)

    if args.ref_path and args.input_path:
        from bagel_infer.pipeline import predict_single_edit

        os.makedirs(args.save_dir, exist_ok=True)
        pred = predict_single_edit(
            model,
            processors.image_processor,
            args.ref_path,
            args.input_path,
            device=args.device,
            fp16=args.fp16,
        )
        tag = getattr(pred, "_debug_tag", "abs_direct")
        base, ext = os.path.splitext(args.out_name)
        out_path = os.path.join(args.save_dir, f"{base}__{tag}{ext}")
        pred.save(out_path)
        print(f"[infer] wrote {out_path}")
        return

    results = run_batch(
        model=model,
        processors=processors,
        dataset_root=args.dataset_root,
        save_dir=args.save_dir,
        device=args.device,
        fp16=args.fp16,
        num_timesteps=args.num_timesteps,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=tuple(args.cfg_interval),
    )
    print(f"[infer] wrote {len(results)} predictions to {os.path.join(args.save_dir, 'preds')}")


if __name__ == "__main__":
    main()
