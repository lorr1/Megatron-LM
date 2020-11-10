#!/bin/bash
TASK="WIKITEXT103"
VALID_DATA=/u/scr/nlp/ooa/megatron-preprocessed-data/wikitext_raw/wikitext-103/wiki.test.tokens
CHECKPOINT_PATH=/u/scr/nlp/ooa/megatron-preprocessed-data/testing_outputs/gpt2_wiki103
TENSORBOARD_PATH=/u/scr/nlp/ooa/megatron-preprocessed-data/testing_outputs/gpt2_wiki103/tensorboard
VOCAB_FILE=/u/scr/nlp/ooa/megatron-preprocessed-data/hugging_face_gpt2/gpt2-vocab.json
MERGE_FILE=/u/scr/nlp/ooa/megatron-preprocessed-data/hugging_face_gpt2/gpt2-merges.txt
DATA_PATH=/u/scr/nlp/ooa/megatron-preprocessed-data/wikitext103/wikitext103_text_document

COMMON_TASK_ARGS="--num-layers 4 \
                  --hidden-size 512 \
                  --num-attention-heads 16 \
                  --seq-length 512 \
                  --max-position-embeddings 512 \
                  --fp16"

python tasks/main.py \
       --config configs/base_output.yaml \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPT2BPETokenizer \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --batch-size 8 \
       --checkpoint-activations \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng
