#!/bin/sh
TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=640
UPDATE_FREQ=32
BART_PATH=/home/s1617290/lacus/model/bart.large/model_${MAX_TOKENS}.pt
DATA_BIN=$1
SAVE_DIR=$2

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py $DATA_BIN \
    --task guided_translation \
    --arch guided_bart_large \
    --criterion label_smoothed_cross_entropy \
    --memory-efficient-fp16 \
    --ddp-backend=no_c10d\
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --max-source-positions $MAX_TOKENS \
    --max-target-positions $MAX_TOKENS \
    --truncate-source \
    --source-lang source \
    --target-lang target \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --save-dir $SAVE_DIR \
    --find-unused-parameters;
