import os
import gigapath.slide_encoder as slide_encoder

assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

# load from HuggingFace
# NOTE: CLS token is not trained during the pretraining
model_cls = slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536, global_pool=False)

# load from HuggingFace with global pooling
model_global_pool = slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536, global_pool=True)

# load from local file
# model = slide_encoder.create_model(
#    "/datadrive/gigapath/slide_encoder.pth",
#    "gigapath_slide_enc12l768d",
#    1536,
#)

# directly initialize a model using timm (random init)
# import timm
# model = timm.create_model("gigapath_slide_enc12l768d", pretrained=False, in_chans=1536)

# random init
# model = slide_encoder.create_model("", "gigapath_slide_enc12l768d", 1536)

print("param #", sum(p.numel() for p in model_global_pool.parameters()))
