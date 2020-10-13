#!/bin/bash

s=$4
env=$1
setting=$2
par=$3
suf=${@:5}
run()
{
    if [[ "$env" == "Ant"* ]]; then
        SEEDS=( 13248 16542 17648 25911 30624 )
    elif [[ "$env" == "HalfCheetah"* ]]; then
        SEEDS=( 2533 9950 15509 15696 15737 )
    elif [[ "$env" == "Humanoid"* ]]; then
        SEEDS=( 38675 4975 37471 14464 19509 )
    elif [[ "$env" == "DeepMimic"* ]]; then
        SEEDS=( 34114 33406 28949 12831 39907 )
    elif [[ "$env" == "Reacher"* ]]; then
        SEEDS=( 31852 30901 32503 32593 3718 )
    else
        echo "Unkown Env: ${env}"
        exit
    fi
    seed=${SEEDS[s]}

    cmd="OPENBLAS_NUM_THREADS=1 python main.py --env ${env} --seed ${seed} --setting settings.${setting} --particles ${par} ${suf}"


    echo $cmd
    eval $cmd
}

run $*
