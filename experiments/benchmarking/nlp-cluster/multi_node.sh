
GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost  #"jagupard26.stanford.edu" #localhost
MASTER_PORT=6000
NNODES=2
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR" # --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt2.py \
       --config configs/fixed_args.yaml


       # nlprun -a kgoel-py38-megatronlm -m jagupard10 -g 5 -p high -q jag -w /juice/scr/lorr1/Megatron-LM -s zsh `bash examples/quin_distributed.sh`