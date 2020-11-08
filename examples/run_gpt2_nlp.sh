CHECKPOINT_PATH=/u/scr/nlp/ooa/megatron-preprocessed-data/testing_outputs/gpt2_baby
TENSORBOARD_PATH=/u/scr/nlp/ooa/megatron-preprocessed-data/testing_outputs/gpt2_baby/tensorboard
VOCAB_FILE=/u/scr/nlp/ooa/megatron-preprocessed-data/hugging_face_gpt2/gpt2-vocab.json
MERGE_FILE=/u/scr/nlp/ooa/megatron-preprocessed-data/hugging_face_gpt2/gpt2-merges.txt
DATA_PATH=/u/scr/nlp/ooa/megatron-preprocessed-data/baby_dataset/test_abc_text_document

GPT2_ARGS="--num-layers 1 \
           --hidden-size 64 \
           --num-attention-heads 8 \
           --seq-length 16 \
           --max-position-embeddings 1024 \
           --batch-size 16 \
           --lr 0.00015 \
           --lr-decay-iters 320000 \
           --lr-decay-style cosine \
           --vocab-file $VOCAB_FILE \
           --merge-file $MERGE_FILE \
           --warmup .01 \
           --fp16"

WANDB_ARGS="--group test_run"

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --train-iters 10000 \
             --no-save-optim \
             --no-save-rng \
             --checkpoint-activations"

python pretrain_gpt2.py \
       $GPT2_ARGS \
       $OUTPUT_ARGS \
       $WANDB_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --tensorboard-dir $TENSORBOARD_PATH \
       --split "33, 33, 33"