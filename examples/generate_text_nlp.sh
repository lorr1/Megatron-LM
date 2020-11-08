#!/bin/bash

CHECKPOINT_PATH=/u/scr/nlp/ooa/megatron-preprocessed-data/testing_outputs/gpt2_baby
TENSORBOARD_PATH=/u/scr/nlp/ooa/megatron-preprocessed-data/testing_outputs/gpt2_baby/tensorboard
VOCAB_FILE=/u/scr/nlp/ooa/megatron-preprocessed-data/hugging_face_gpt2/gpt2-vocab.json
MERGE_FILE=/u/scr/nlp/ooa/megatron-preprocessed-data/hugging_face_gpt2/gpt2-merges.txt
DATA_PATH=/u/scr/nlp/ooa/megatron-preprocessed-data/baby-dataset2/my-gpt2_text_document


python tools/generate_samples_gpt2.py \
       --model-parallel-size 1 \
       --num-layers 1 \
       --hidden-size 64 \
       --load $CHECKPOINT_PATH \
       --num-attention-heads 2 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --batch-size 2 \
       --seq-length 4 \
       --out-seq-length 4 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --genfile unconditional_samples.json \
       --num-samples 20 \
       --top_p 0.9 \
       --no-load-optim \
       --no-load-rng \
       --recompute
