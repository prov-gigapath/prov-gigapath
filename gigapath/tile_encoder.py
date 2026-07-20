# --------------------------------------------------------
# GigaPath-Flash tile encoder
# --------------------------------------------------------
# The GigaPath-Flash tile encoder is a DINOv2-small backbone: a ViT-S/16
# (384-dim, 12 layers, 6 heads) with SwiGLU FFN and LayerScale. It produces the
# 384-dim [CLS] embedding consumed by the GigaPath-Flash slide encoder
# (gigapath_slide_enc12l384d).
#
# The released weights are packaged in the timm / HuggingFace style
# (config.json + pytorch_model.bin), mirroring the original Prov-GigaPath tile
# encoder. Raw DINOv2 student checkpoints can be converted with
# scripts/convert_tile_encoder_checkpoint.py.

import os

import torch
import torch.nn as nn

import timm
import huggingface_hub
from timm.models.registry import register_model
from timm.layers import SwiGLUPacked
from timm.models.vision_transformer import VisionTransformer

# ViT-S/16 SwiGLU (DINOv2-small) architecture arguments.
_TILE_ENC_ARGS = dict(
    img_size=224,
    patch_size=16,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=2048 / 384.0,  # SwiGLU: fc1 -> 2048, fc2 <- 1024 (matches DINOv2 w12/w3)
    mlp_layer=SwiGLUPacked,
    act_layer=nn.SiLU,
    init_values=1e-5,  # LayerScale
    num_classes=0,
    global_pool="token",
    class_token=True,
    reg_tokens=0,
)


@register_model
def gigapath_tile_enc_dinov2s(pretrained=False, **kwargs):
    """GigaPath-Flash tile encoder (DINOv2-small ViT-S/16, SwiGLU, LayerScale)."""
    model_args = dict(_TILE_ENC_ARGS)
    # timm may inject a `pretrained_cfg`/`pretrained_cfg_overlay`; VisionTransformer
    # does not accept those, so drop them here.
    for k in ("pretrained_cfg", "pretrained_cfg_overlay", "cache_dir"):
        kwargs.pop(k, None)
    model_args.update(kwargs)
    return VisionTransformer(**model_args)


def create_model(
    pretrained: str,
    model_arch: str = "gigapath_tile_enc_dinov2s",
    local_dir: str = os.path.join(os.path.expanduser("~"), ".cache/"),
    **kwargs,
):
    """Build the GigaPath-Flash tile encoder and optionally load pretrained weights.

    Arguments:
    ----------
    pretrained: str
        Either a local path to a timm-style checkpoint (pytorch_model.bin) or a
        "hf_hub:<org>/<repo>" identifier to download it from the HuggingFace Hub.
    model_arch: str
        The registered tile-encoder architecture name.
    """
    model = timm.create_model(model_arch, pretrained=False, **kwargs)

    if pretrained.startswith("hf_hub:"):
        hub_name = pretrained.split(":")[1]
        huggingface_hub.hf_hub_download(hub_name, filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
        local_path = os.path.join(local_dir, "pytorch_model.bin")
    else:
        local_path = pretrained

    if os.path.exists(local_path):
        state_dict = torch.load(local_path, map_location="cpu")
        # accept both a bare state_dict and a {"model": state_dict} wrapper
        if isinstance(state_dict, dict) and "patch_embed.proj.weight" not in state_dict and "model" in state_dict:
            state_dict = state_dict["model"]

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        for k in missing_keys:
            print("Missing ", k)
        for k in unexpected_keys:
            print("Unexpected ", k)
        print("\033[92m Successfully Loaded Pretrained GigaPath-Flash tile encoder from {} \033[00m".format(pretrained))
    else:
        print("\033[93m Pretrained weights not found at {}. Randomly initialized the model! \033[00m".format(local_path))

    return model
