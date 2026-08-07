from gigapath.pipeline import tile_one_slide
import huggingface_hub
import os

homedir_path = os.path.expanduser("~")
assert ("HF_TOKEN" in os.environ) or os.path.exists(f"{homedir_path}/.cache/huggingface/token"), "Please set the HF_TOKEN environment variable to your Hugging Face API token or make sure the token is cached in ~/.cache/huggingface/token"

local_dir = "./sample_data/"
slide_path = os.path.join(local_dir, "PROV-000-000001.ndpi")
huggingface_hub.hf_hub_download("prov-gigapath/prov-gigapath", filename=slide_path, local_dir=".", force_download=True)

save_dir = os.path.join(local_dir, 'outputs/preprocessing/')

print("NOTE: Prov-GigaPath is trained with 0.5 mpp preprocessed slides. Please make sure to use the appropriate level for the 0.5 MPP")
tile_one_slide(slide_path, save_dir=save_dir, level=1)

print("NOTE: tiling dependency libraries can be tricky to set up. Please double check the generated tile images.")
