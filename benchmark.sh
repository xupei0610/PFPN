#!/bin/bash

s=$4
env=$1
setting=$2
par=$3
suf=${@:5}
run()
{
    if [[ "$env" == "DeepMimic"* ]]; then
        SEEDS=( 34114 33406 28949 12831 39907 )
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
