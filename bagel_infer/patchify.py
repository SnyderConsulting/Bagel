import torch


def unpatchify_v1(x: torch.Tensor, p: int) -> torch.Tensor:
    """Convert [B, C*(p*p), H, W] -> [B, C, H*p, W*p] using the standard layout."""

    B, Cpp, H, W = x.shape
    C = Cpp // (p * p)
    x = x.view(B, C, p, p, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(B, C, H * p, W * p)


def unpatchify_v2(x: torch.Tensor, p: int) -> torch.Tensor:
    """Alternative axis order used by some repositories."""

    B, Cpp, H, W = x.shape
    C = Cpp // (p * p)
    x = x.view(B, p, p, C, H, W)
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
    return x.view(B, C, H * p, W * p)
