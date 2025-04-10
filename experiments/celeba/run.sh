mkdir -p ./save
mkdir -p ./trainlogs
mkdir -p ./trainlogs/losses

method=famo
seed=42
gamma=0.001

python trainer.py --method=$method --seed=$seed --gamma=$gamma > trainlogs/log_$method-gamma$gamma-$seed.log 2>&1 &
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