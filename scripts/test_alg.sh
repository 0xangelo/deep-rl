#!/bin/sh

alg=$1
task=$2
envs=$3
log_dir=data/test/$task/$alg
seeds=(471203247) # 1439988764 1708339277 3461167908 546083439 1826430695 205518236 1911719948 3685968396 796215862)
rm -rf $log_dir*

if [ $alg == "vanilla" ] || [ $alg == "trpo" ] || [ $alg == "ppo" ] || [ $alg == "acktr" ]
then
    python -m proj.run $alg --log_dir $log_dir --exp_name $task-$alg --env $task --policy:class MlpPolicy --steps 'int(1e6)' --n_envs $envs --seed ${seeds[*]}
fi
if [ $alg == "a2c" ]
then
    python -m proj.run $alg --log_dir $log_dir --exp_name $task-$alg --env $task --policy:class CNNWeightSharingAC --samples 'int(1e6)' --n_envs $envs --optimizer:lr 1e-3 --seed ${seeds[*]}
fi
