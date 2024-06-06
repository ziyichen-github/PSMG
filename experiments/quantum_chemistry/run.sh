mkdir -p ./save
mkdir -p ./trainlogs

method=fairgrad
seed=42
gamma=0.001
lr=0.001
bs=120
alpha=2.0
# Get the current time in hhmmss format
timestamp=$(date +"%H%M%S")
# For fairgrad only
python trainer.py --method=$method --seed=$seed --gamma=$gamma --alpha=$alpha --scale-y=True --lr=$lr --batch-size=$bs > trainlogs/log_${method}-g${gamma}-${seed}-lr${lr}-bs${bs}-${timestamp}.log 2>&1 &
# python trainer.py --method=$method --seed=$seed --gamma=$gamma --scale-y=True --lr=$lr --batch-size=$bs > trainlogs/log_${method}-g${gamma}-${seed}-lr${lr}-bs${bs}-${timestamp}.log 2>&1 &
# python trainer.py --method=$method --seed=$seed --gamma=$gamma --scale-y=True --lr=$lr --batch-size=$bs > trainlogs/log_$method-g$gamma-$seed-lr$lr-bs$bs.log 2>&1 &
# stat -c %A /home/mx6835/.dirnews
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
#     fairgrad=FairGrad,
# )
