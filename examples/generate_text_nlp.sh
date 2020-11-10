#!/bin/bash

CHECKPOINT_PATH=/u/scr/nlp/ooa/megatron-preprocessed-data/testing_outputs/gpt2_wiki103
TENSORBOARD_PATH=/u/scr/nlp/ooa/megatron-preprocessed-data/testing_outputs/gpt2_wiki103/tensorboard
VOCAB_FILE=/u/scr/nlp/ooa/megatron-preprocessed-data/hugging_face_gpt2/gpt2-vocab.json
MERGE_FILE=/u/scr/nlp/ooa/megatron-preprocessed-data/hugging_face_gpt2/gpt2-merges.txt
DATA_PATH=/u/scr/nlp/ooa/megatron-preprocessed-data/wikitext103/wikitext103_text_document


python tools/generate_samples_gpt2.py \
       --model-parallel-size 5 \
       --num-layers 4 \
       --hidden-size 512 \
       --load $CHECKPOINT_PATH \
       --num-attention-heads 16 \
       --max-position-embeddings 512 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --batch-size 2 \
       --seq-length 512 \
       --out-seq-length 512 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --genfile unconditional_samples.json \
       --num-samples 0 \
       --top_p 0.9 \
       --no-load-optim \
       --no-load-rng \
       --recompute
