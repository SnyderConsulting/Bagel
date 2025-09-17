import torch


def vae_scaling_factor(vae) -> float:
    for path in ("scaling_factor",):
        if hasattr(vae, path):
            return float(getattr(vae, path))
    cfg = getattr(vae, "config", None)
    if cfg is not None and hasattr(cfg, "scaling_factor"):
        return float(cfg.scaling_factor)
    return 1.0


@torch.no_grad()
def vae_encode(vae, x: torch.Tensor) -> torch.Tensor:
    """
    Encode an image tensor into VAE latents scaled by the VAE's scaling factor.

    Args:
        vae: Autoencoder model with an ``encode`` method.
        x: Tensor of shape [B, 3, H, W] typically in [-1, 1] or [0, 1].

    Returns:
        Latent tensor [B, C, H', W'] already multiplied by the scaling factor so
        that ``vae.decode(latent)`` reconstructs the input directly.
    """

    e = vae.encode(x)
    if hasattr(e, "latent_dist"):
        z = e.latent_dist.sample()
    elif hasattr(e, "sample"):
        z = e.sample()
    elif hasattr(e, "latents"):
        z = e.latents
    else:
        z = e
    sf = vae_scaling_factor(vae)
    if sf != 1.0:
        z = z * sf
    return z


@torch.no_grad()
def vae_decode(vae, z: torch.Tensor) -> torch.Tensor:
    """
    Decode latents into pixel space, respecting the VAE scaling factor.

    Args:
        vae: Autoencoder model with a ``decode`` method.
        z: Latent tensor already scaled (see :func:`vae_encode`).

    Returns:
        Tensor [B, 3, H, W] in [0, 1].
    """

    sf = vae_scaling_factor(vae)
    if sf != 1.0:
        z = z / sf
    img = vae.decode(z)
    if img.min() < 0:
        img = (img.clamp(-1, 1) + 1) / 2
    return img.clamp(0, 1)
