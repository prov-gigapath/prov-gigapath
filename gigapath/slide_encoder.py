# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
#
# Portions Copyright Prov-GigaPath
# Original File: https://github.com/facebookresearch/mae

from functools import partial

import os
import sys
import torch
import torch.nn as nn
import numpy as np

import timm
from timm.models.registry import register_model
import huggingface_hub

from .pos_embed import get_2d_sincos_pos_embed
from .torchscale.model.LongNet import make_longnet_from_name


class PatchEmbed(nn.Module):
    """Slide Patch Embedding"""

    def __init__(
        self,
        in_chans=1536,
        embed_dim=768,
        norm_layer=None,
        bias=True,
    ):
        super().__init__()

        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, L, D = x.shape
        x = self.proj(x)
        x = self.norm(x)
        return x


class LongNetViT(nn.Module):
    """
    Backbone of Vision Transformer for downstream tasks

    Arguments:
    ----------
    in_chans: int
        The number of input channels, should be the tile encoding dimension 1536.
    embed_dim: int
        The embedding dimension of the LongNet model.
    depth: int
        The number of LongNet layers in the LongNet model.
    slide_ngrids: int
        The number of grids in the slide.
    tile_size: int
        The tile size. Default is 256px.
    max_wsi_size: int
        The maximum size of the WSI.
    norm_layer: nn.LayerNorm
        The normalization layer used in the model.
    global_pool: bool
        Whether to use global pooling or not.
    dropout: float
        The dropout rate used in the model.
    drop_path_rate: float
        The drop path rate used in the model.
    """

    def __init__(self, 
                in_chans=1536, 
                embed_dim=256, 
                depth=12, 
                slide_ngrids=1000, 
                tile_size=256,
                max_wsi_size=262144,
                norm_layer=nn.LayerNorm, 
                global_pool=False, 
                dropout=0.25, 
                drop_path_rate=0.1, 
                **kwargs):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(in_chans, embed_dim)
        
        self.tile_size = tile_size
        self.slide_ngrids = slide_ngrids
        num_patches = slide_ngrids**2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer('pos_embed', torch.zeros(1, num_patches + 1, embed_dim), persistent=False)  # fixed sin-cos embedding

        self.encoder_name = "LongNet_{}_layers_{}_dim".format(depth, embed_dim)
        if kwargs.get("mlp_ratio", 4.0) != 4.0:
            self.encoder_name += "_mlp{}".format(kwargs.get("mlp_ratio"))
        
        # get optimal segment length
        segment_length = self.get_optimal_segment_length(max_wsi_size, tile_size)
        self.encoder = make_longnet_from_name(self.encoder_name, drop_path_rate=drop_path_rate, dropout=dropout, segment_length=segment_length)
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.global_pool = global_pool
        print("Global Pooling:", self.global_pool)

        self.initialize_vit_weights()

    def initialize_vit_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.slide_ngrids, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def get_optimal_segment_length(self, max_wsi_size: int=262144, tile_size: int=256) -> str:
        '''
        Get the optimal segment length based on the maximum image size and tile size.
        
        Arguments:
        ----------
        max_wsi_size: int
            The maximum size of the WSI.
        tile_size: int
            The tile size.
        '''
        max_seq_len = (max_wsi_size // tile_size) ** 2
        # calculate the segment length
        segment_length = np.linspace(np.log2(1024), int(np.log2(max_seq_len)), 5)
        segment_length = np.power(2, segment_length).astype(int)
        # convert to str format
        segment_length = str(list(segment_length))
        return segment_length

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def coords_to_pos(self, coords, tile_size: int = 256):
        """
        This function is used to convert the coordinates to the positional indices

        Arguments:
        ----------
        coords: torch.Tensor
            The coordinates of the patches, of shape [N, L, 2]
        output: torch.Tensor
            The positional indices of the patches, of shape [N, L]
        """
        coords_ = torch.floor(coords / tile_size)
        pos = coords_[..., 0] * self.slide_ngrids + coords_[..., 1]
        return pos.long() + 1  # add 1 for the cls token

    def forward(self, x, coords, all_layer_embed=False):
        """
        The forward pass of the model

        Arguments:
        ----------
        x: torch.Tensor
            The input tile embeddings, of shape [N, L, D]
        coords: torch.Tensor
            The coordinates of the patches, of shape [N, L, 2]
        all_layer_embed: bool
            Whether to return embeddings from all layers or not
        """
        # embed patches
        x = self.patch_embed(x)

        # get pos indices
        pos = self.coords_to_pos(coords, self.tile_size)  # [N, L]

        x = x + self.pos_embed[:, pos, :].squeeze(0)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        if all_layer_embed:
            x_list = self.encoder(src_tokens=None, token_embeddings=x, return_all_hiddens=all_layer_embed)["encoder_states"]
        else:
            x_list = [self.encoder(src_tokens=None, token_embeddings=x)["encoder_out"]]

        outcomes = []
        for x in x_list:
            if self.global_pool:
                x = x[:, 1:, :].mean(dim=1)  # global average pooling
                outcome = self.norm(x)
            else:
                x = self.norm(x)
                outcome = x[:, 0]
            outcomes.append(outcome)

        return outcomes


def create_model(pretrained: str, model_arch: str, in_chans: int, local_dir: str = os.path.join(os.path.expanduser("~"), ".cache/"), **kwargs):
    model = timm.create_model(model_arch, pretrained=False, in_chans=in_chans, **kwargs)

    if pretrained.startswith("hf_hub:"):
        hub_name = pretrained.split(":")[1]
        huggingface_hub.hf_hub_download(hub_name, filename="slide_encoder.pth", local_dir=local_dir, force_download=True)
        local_path = os.path.join(local_dir, "slide_encoder.pth")
    else:
        local_path = pretrained

    if os.path.exists(local_path):
        state_dict = torch.load(local_path, map_location="cpu")["model"]

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if len(missing_keys) > 0:
            for k in missing_keys:
                print("Missing ", k)

        if len(unexpected_keys) > 0:
            for k in unexpected_keys:
                print("Unexpected ", k)

        print("\033[92m Successfully Loaded Pretrained GigaPath model from {} \033[00m".format(pretrained))
    else:
        print("\033[93m Pretrained weights not found at {}. Randomly initialized the model! \033[00m".format(local_path))

    return model


@register_model
def gigapath_slide_enc12l768d(**kwargs):
    model = LongNetViT(embed_dim=768, depth=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def gigapath_slide_enc24l1024d(**kwargs):
    model = LongNetViT(embed_dim=1024, depth=24, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def gigapath_slide_enc12l1536d(**kwargs):
    model = LongNetViT(embed_dim=1536, depth=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
