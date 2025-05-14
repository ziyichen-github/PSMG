mkdir -p ./save
mkdir -p ./trainlogs
mkdir -p ./trainlogs/losses

method=pmgd
seed=42
gamma=0.001
n_epochs=200
batch_size=16
num_agents=5
network_sparsity=0.6

python trainer.py --method=$method --seed=$seed --gamma=$gamma --n_epochs=$n_epochs --batch_size=$batch_size --num_agents=$num_agents --network_sparsity=$network_sparsity > trainlogs/log_$method-gamma$gamma-$seed-batch$batch_size-agent$num_agents-sparsity$network_sparsity.log 2>&1 &
# method=sdmgrad
# seed=42
# lamda=0.3
# niter=20

# python -u trainer.py --method=$method --seed=$seed --lamda=$lamda --niter=$niter > trainlogs/sdmgrad-lambda$lamda-sd$seed.log 2>&1 &
# METHODS = dict(
#     stl=STL,
#     ls=LinearScalarization,
#     uw=Uncertainty,
#     scaleinvls=ScaleInvariantLinearScalarization,
#     rlw=RLW,
#     dwa=DynamicWeightAverage,
#     pcgrad=PCGrad,
#     mgda=MGDA,
#     graddrop=GradDrop,
#     log_mgda=LOG_MGDA,
#     cagrad=CAGrad,
#     log_cagrad=LOG_CAGrad,
#     imtl=IMTLG,
#     log_imtl=LOG_IMTLG,
#     nashmtl=NashMTL,
#     famo=FAMO,
#     pmgdn=PMGDN,
#     pmgdnlog=PMGDNLog,
#     pmgdlog1=PMGDLog1,
# )