#!/bin/bash

# Common parameters
method=pmgd
seed=42
gamma=0.001
lr=1e-4
n_epochs=2
num_agents=5
network_sparsity=0.6

# Run experiments in parallel (4 jobs)
parallel -j4 \
  'python trainer.py \
    --method={1} \
    --seed={2} \
    --gamma={3} \
    --lr={4} \
    --n_epochs={5} \
    --batch_size={6} \
    --num_agents={7} \
    --network_sparsity={8} \
    > trainlogs/log_{1}-gamma{3}-{2}-batch{6}-agent{7}-sparsity{8}.log 2>&1' \
  ::: $method ::: $seed ::: $gamma ::: $lr ::: $n_epochs \
  ::: 1 2 4 8 \          # Batch sizes for experiments 1-4
  ::: $num_agents ::: $network_sparsity