# Finetune GigaPath-Flash (mini) slide encoder on PANDA.
#
# GigaPath-Flash: LongNet 12-layer / 384-dim slide encoder paired with a
# 384-dim tile encoder (DINOv2-small). Both the slide encoder hidden dim and the
# input tile-embedding dim are 384 (vs. 768/1536 for the original GigaPath).
#
# Provide a converted slide-encoder checkpoint via GIGAPATH_FLASH_CKPT (produced
# with scripts/convert_slide_encoder_checkpoint.py), or a hf_hub:... path.

# Task setting
TASKCFG=finetune/task_configs/panda.yaml
DATASETCSV=dataset_csv/PANDA/PANDA.csv
PRESPLITDIR=dataset_csv/PANDA/ # Use the predefined split
ROOTPATH=${1-:data/dinov2_features/h5_files}
MAX_WSI_SIZE=250000  # Maximum WSI size in pixels for the longer side (width or height).
TILE_SIZE=256
# Model settings (GigaPath-Flash / mini: LongNet 12 layers, 384 dim)
HFMODEL=${GIGAPATH_FLASH_CKPT:-hf_hub:prov-gigapath/prov-gigapath-flash}
MODELARCH=gigapath_slide_enc12l384d
TILEEMBEDSIZE=384
LATENTDIM=384
# Training settings
EPOCH=5
GC=32
BLR=0.002
WD=0.05
LD=0.95
FEATLAYER="11"
DROPOUT=0.1
# Output settings
WORKSPACE=outputs/PANDA_flash
SAVEDIR=$WORKSPACE
EXPNAME=run_flash_epoch-${EPOCH}_blr-${BLR}_wd-${WD}_ld-${LD}_feat-${FEATLAYER}

echo "Data directory set to $ROOTPATH"

python finetune/main.py --task_cfg_path ${TASKCFG} \
               --dataset_csv $DATASETCSV \
               --root_path $ROOTPATH \
               --model_arch $MODELARCH \
               --blr $BLR \
               --layer_decay $LD \
               --optim_wd $WD \
               --dropout $DROPOUT \
               --drop_path_rate 0.0 \
               --val_r 0.1 \
               --epochs $EPOCH \
               --input_dim $TILEEMBEDSIZE \
               --latent_dim $LATENTDIM \
               --feat_layer $FEATLAYER \
               --warmup_epochs 1 \
               --gc $GC \
               --model_select last_epoch \
               --lr_scheduler cosine \
               --folds 1 \
               --dataset_csv $DATASETCSV \
               --pre_split_dir $PRESPLITDIR \
               --save_dir $SAVEDIR \
               --pretrained $HFMODEL \
               --report_to tensorboard \
               --exp_name $EXPNAME \
               --max_wsi_size $MAX_WSI_SIZE
