from gigapath.pipeline import tile_one_slide
import huggingface_hub
import os

assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

local_dir = os.path.join(os.path.expanduser("~"), ".cache/")
huggingface_hub.hf_hub_download("prov-gigapath/prov-gigapath", filename="sample_data/PROV-000-000001.ndpi", local_dir=local_dir, force_download=True)
slide_path = os.path.join(local_dir, "sample_data/PROV-000-000001.ndpi")

save_dir = os.path.join(local_dir, 'outputs/preprocessing/')

print("NOTE: Prov-GigaPath is trained with 0.5 mpp preprocessed slides. Please make sure to use the appropriate level for the 0.5 MPP")
tile_one_slide(slide_path, save_dir=save_dir, level=1)

print("NOTE: tiling dependency libraries can be tricky to set up. Please double check the generated tile images.")
