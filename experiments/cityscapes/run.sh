mkdir -p ./save
mkdir -p ./trainlogs

# method=sdmgrad
seed=42
gamma=0.01

method=sdmgrad
# seed=0
lamda=0.3
niter=20

# python trainer.py --method=$method --seed=$seed --gamma=$gamma > trainlogs/log_$method-gamma$gamma-$seed.log 2>&1 &
python -u trainer.py --method=$method --seed=$seed --lamda=$lamda --niter=$niter > trainlogs/sdmgrad-lambda$lamda-sd$seed.log 2>&1 &
# METHODS = dict(
    # stl=STL,
    # ls=LinearScalarization,
    # uw=Uncertainty,
    # scaleinvls=ScaleInvariantLinearScalarization,
    # rlw=RLW,
    # dwa=DynamicWeightAverage,
    # pcgrad=PCGrad,
    # mgda=MGDA,
    # graddrop=GradDrop,
    # log_mgda=LOG_MGDA,
    # cagrad=CAGrad,
    # log_cagrad=LOG_CAGrad,
    # imtl=IMTLG,
    # log_imtl=LOG_IMTLG,
    # nashmtl=NashMTL,
    # famo=FAMO,
    # pmgdn=PMGDN,
    # pmgdnlog=PMGDNLog,
    # pmgdlog1=PMGDLog1,
    # fairgrad=FairGrad,
    # sdmgrad=SDMGrad,
# )