mkdir -p ./save
mkdir -p ./trainlogs
mkdir -p ./trainlogs/losses

method=pmgd
seed=42
gamma=0.001
n_epochs=200
batch_size=2
num_agents=5
network_sparsity=0.6

python trainer.py --method=$method --seed=$seed --gamma=$gamma --n_epochs=$n_epochs --batch_size=$batch_size --num_agents=$num_agents --network_sparsity=$network_sparsity > trainlogs/log_$method-gamma$gamma-$seed-batch$batch_size-agent$num_agents-sparsity$network_sparsity.log 2>&1 &