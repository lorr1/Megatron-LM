CHECKPOINT_PATH=/u/scr/nlp/ooa/megatron-preprocessed-data/testing_outputs/gpt2_wiki103
TENSORBOARD_PATH=/u/scr/nlp/ooa/megatron-preprocessed-data/testing_outputs/gpt2_wiki103/tensorboard
VOCAB_FILE=/u/scr/nlp/ooa/megatron-preprocessed-data/hugging_face_gpt2/gpt2-vocab.json
MERGE_FILE=/u/scr/nlp/ooa/megatron-preprocessed-data/hugging_face_gpt2/gpt2-merges.txt
DATA_PATH=/u/scr/nlp/ooa/megatron-preprocessed-data/wikitext103/wikitext103_text_document

GPUS_PER_NODE=5
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

GPT2_ARGS="--num-layers 4 \
           --hidden-size 512 \
           --num-attention-heads 16 \
           --seq-length 512 \
           --max-position-embeddings 512 \
           --batch-size 56 \
           --lr 0.00015 \
           --distributed-backend nccl \
           --lr-decay-iters 320000 \
           --lr-decay-style cosine \
           --vocab-file $VOCAB_FILE \
           --merge-file $MERGE_FILE \
           --warmup .01 \
           --fp16"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

WANDB_ARGS="--group test_run"

OUTPUT_ARGS="--log-interval 20 \
             --save-interval 1000 \
             --eval-interval 500 \
             --eval-iters 10 \
             --train-iters 10000 \
             --no-save-optim \
             --no-save-rng \
             --checkpoint-activations"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt2.py \
       $GPT2_ARGS \
       $OUTPUT_ARGS \
       $WANDB_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --tensorboard-dir $TENSORBOARD_PATH \
       --split "100, 10, 10"


       # nlprun -a kgoel-py38-megatronlm -m jagupard10 -g 5 -p high -q jag -w /juice/scr/lorr1/Megatron-LM -s zsh `bash examples/run_gpt2_nlp_distributed.sh`