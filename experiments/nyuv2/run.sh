mkdir -p ./save
mkdir -p ./trainlogs
mkdir -p ./trainlogs/losses

method=pmgd
seed=42
gamma=0.001
batch_size=5

python trainer.py --method=$method --seed=$seed --gamma=$gamma --batch_size=$batch_size > trainlogs/log_$method-gamma$gamma-$seed-$batch_size.log 2>&1
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