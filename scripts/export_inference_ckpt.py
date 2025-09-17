"""Export a consolidated Bagel inference checkpoint."""
import os

from safetensors.torch import save_file

from bagel_infer.factory import build_model, load_checkpoint


def main() -> None:
    out = os.environ.get("OUT", "inference.safetensors")
    model = build_model(
        model_path="/workspace/Bagel/models/BAGEL-7B-MoT",
        llm_path="/workspace/models/Qwen2.5-0.5B-Instruct",
        vit_path="/workspace/models/siglip-so400m-14-980-flash-attn2-navit",
        vae_path="/workspace/Bagel/flux/vae/ae.safetensors",
        device="cpu",
        max_latent_size=64,
        latent_patch_size=2,
        vit_max_num_patch_per_side=30,
    )
    load_checkpoint(
        model,
        "/workspace/Bagel/results/breast_edit_min/checkpoints/0002000",
        prefer_ema=True,
    )
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
