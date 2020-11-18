
GPUS_PER_NODE=5
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt2.py \
       --config configs/pretrain_gpt2_wikitext.yaml


       # nlprun -a kgoel-py38-megatronlm -m jagupard10 -g 5 -p high -q jag -w /juice/scr/lorr1/Megatron-LM -s zsh `bash examples/run_gpt2_nlp_distributed.sh`