import huggingface_hub
import os
from gigapath.preprocessing.data.slide_utils import find_level_for_target_mpp

homedir_path = os.path.expanduser("~")
assert ("HF_TOKEN" in os.environ) or os.path.exists(f"{homedir_path}/.cache/huggingface/token"), "Please set the HF_TOKEN environment variable to your Hugging Face API token or make sure the token is cached in ~/.cache/huggingface/token"

local_dir = "./sample_data/"
slide_path = os.path.join(local_dir, "PROV-000-000001.ndpi")
huggingface_hub.hf_hub_download("prov-gigapath/prov-gigapath", filename=slide_path, local_dir=".", force_download=True)

print("NOTE: Prov-GigaPath is trained with 0.5 mpp preprocessed slides")
target_mpp = 0.5
level = find_level_for_target_mpp(slide_path, target_mpp)
if level is not None:
    print(f"Found level: {level}")
else:
    print("No suitable level found.")
