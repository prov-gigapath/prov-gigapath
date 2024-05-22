INPUTPATH=${1-:data/GigaPath_PCam_embeddings.zip}
DATASETCSV=dataset_csv/pcam/pcam.csv
OUTPUT=outputs/pcam

python linear_probe/main.py --input_path $INPUTPATH \
                --dataset_csv $DATASETCSV \
                --output $OUTPUT \
                --batch_size 128 \
                --lr 0.02 \
                --min_lr 0.0 \
                --train_iters 4000 \
                --eval_interval 100 \
                --optim sgd \
                --weight_decay 0.01 \
