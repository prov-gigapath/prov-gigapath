import timm
from PIL import Image
from torchvision import transforms
import torch
import os

assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

print("param #", sum(p.numel() for p in tile_encoder.parameters()))

transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

img_path = "images/prov_normal_000_1.png"
sample_input = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)

with torch.no_grad():
    output = tile_encoder(sample_input).squeeze()
    print("Model output:", output.shape)
    print(output)

expected_output = torch.load("images/prov_normal_000_1.pt")
print("Expected output:", expected_output.shape)
print(expected_output)

assert torch.allclose(output, expected_output, atol=1e-2)
