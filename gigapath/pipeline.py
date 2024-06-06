# --------------------------------------------------------
# Pipeline for running with GigaPath
# --------------------------------------------------------
import os
import timm
import torch
import numpy as np
import gigapath.slide_encoder as slide_encoder

from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from typing import List, Tuple, Union
from torch.utils.data import Dataset, DataLoader


class TileEncodingDataset(Dataset):
    """
    Do encoding for tiles

    Arguments:
    ----------
    image_paths : List[str]
        List of image paths, each image is named with its coordinates
        Example: ['images/256x_256y.png', 'images/256x_512y.png']
    transform : torchvision.transforms.Compose
        Transform to apply to each image
    """

    def __init__(self, image_paths: List[str], transform=None):
        self.transform = transform
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        # get x, y coordinates from the image name
        x, y = img_name.split(".png")[0].split("_")
        x, y = int(x.replace("x", "")), int(y.replace("y", ""))
        # load the image
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.transform:
                img = self.transform(img)
        return {"img": torch.from_numpy(np.array(img)), "coords": torch.from_numpy(np.array([x, y])).float()}


def load_tile_encoder_transforms() -> transforms.Compose:
    """Load the transforms for the tile encoder"""
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform


def load_tile_slide_encoder(local_tile_encoder_path: str = "", local_slide_encoder_path: str = "") -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load the GigaPath tile and slide encoder models.
    Note: Older versions of timm have compatibility issues. 
    Please ensure that you use a newer version by running the following command: pip install timm>=1.0.3.
    """
    if local_tile_encoder_path:
        tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False, checkpoint_path=local_tile_encoder_path)
    else:
        tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    print("Tile encoder param #", sum(p.numel() for p in tile_encoder.parameters()))

    if local_slide_encoder_path:
        slide_encoder_model = slide_encoder.create_model(local_slide_encoder_path, "gigapath_slide_enc12l768d", 1536)
    else:
        slide_encoder_model = slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536)
    print("Slide encoder param #", sum(p.numel() for p in slide_encoder_model.parameters()))

    return tile_encoder, slide_encoder_model


@torch.no_grad()
def run_inference_with_tile_encoder(image_paths: List[str], tile_encoder: torch.nn.Module, batch_size: int = 128) -> dict:
    """
    Run inference with the tile encoder

    Arguments:
    ----------
    image_paths : List[str]
        List of image paths, each image is named with its coordinates
    tile_encoder : torch.nn.Module
        Tile encoder model
    """
    tile_encoder = tile_encoder.cuda()
    # make the tile dataloader
    tile_dl = DataLoader(TileEncodingDataset(image_paths, transform=load_tile_encoder_transforms()), batch_size=batch_size, shuffle=False)
    # run inference
    tile_encoder.eval()
    collated_outputs = {"tile_embeds": [], "coords": []}
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for batch in tqdm(tile_dl, desc="Running inference with tile encoder"):
            collated_outputs["tile_embeds"].append(tile_encoder(batch["img"].cuda()).detach().cpu())
            collated_outputs["coords"].append(batch["coords"])
    return {k: torch.cat(v) for k, v in collated_outputs.items()}


@torch.no_grad()
def run_inference_with_slide_encoder(tile_embeds: torch.Tensor, coords: torch.Tensor, slide_encoder_model: torch.nn.Module) -> torch.Tensor:
    """
    Run inference with the slide encoder

    Arguments:
    ----------
    tile_embeds : torch.Tensor
        Tile embeddings
    coords : torch.Tensor
        Coordinates of the tiles
    slide_encoder_model : torch.nn.Module
        Slide encoder model
    """
    if len(tile_embeds.shape) == 2:
        tile_embeds = tile_embeds.unsqueeze(0)
        coords = coords.unsqueeze(0)

    slide_encoder_model = slide_encoder_model.cuda()
    slide_encoder_model.eval()
    # run inference
    with torch.cuda.amp.autocast(dtype=torch.float16):
        slide_embeds = slide_encoder_model(tile_embeds.cuda(), coords.cuda(), all_layer_embed=True)
    outputs = {"layer_{}_embed".format(i): slide_embeds[i].cpu() for i in range(len(slide_embeds))}
    outputs["last_layer_embed"] = slide_embeds[-1].cpu()
    return outputs
