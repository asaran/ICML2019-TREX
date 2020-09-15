#!/usr/bin/env bash

for trial in ppo_models/KL/* ; do
    #echo "$trial"
    for c in $trial/checkpoints/*; do
        #echo "inside"
        #echo "$c"
        if [ ${c: -5} == "43000" ]; then
            echo "$c"
        fi
    done
done
