import gigapath.slide_encoder as slide_encoder

# load from HuggingFace
model = slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536)

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

print("param #", sum(p.numel() for p in model.parameters()))
