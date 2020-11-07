CHECKPOINT_PATH=/u/scr/nlp/ooa/megatron-preprocessed-data/testing_outputs/gpt2_345m
VOCAB_FILE=models/gpt2/gpt2-vocab.json
MERGE_FILE=models/gpt2/gpt2-merges.txt
DATA_PATH=/u/scr/nlp/ooa/megatron-preprocessed-data/wiki-en-full/my-gpt2_text_document

GPT2_ARGS="--num-layers 24 \
           --hidden-size 1024 \
           --num-attention-heads 16 \
           --seq-length 1024 \
           --max-position-embeddings 1024 \
           --batch-size 4 \
           --lr 0.00015 \
           --lr-decay-iters 320000 \
           --lr-decay-style cosine \
           --vocab-file $VOCAB_FILE \
           --merge-file $MERGE_FILE \
           --warmup .01 \
           --fp16"

OUTPUT_ARGS="--log-interval 500 \
             --save-interval 6000000 \
             --eval-interval 1000000 \
             --eval-iters 1000 \
             --train-iters 10000000 \
             --no-save-optim \
             --no-save-rng \
             --checkpoint-activations"

python pretrain_gpt2.py \
       $GPT2_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
