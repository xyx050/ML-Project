#!/bin/bash
datasets=(bace bbbp hiv clintox muv sider toxcast tox21 pcba) 
len=${#datasets[@]}
dims=(0 2 5 8)
for((i=0;i<${len};i++));do
    for dim in "${dims[@]}";do
        echo processing "${datasets[${i}]}"--SVM 
        python train.py \
        --data-root ./unimol-2 \
        --data "${datasets[${i}]}" \
        --model-type SVM \
        --dim "${dim}"

        echo processing "${datasets[${i}]}"--RandomForest 
        python train.py \
        --data-root ./unimol-2 \
        --data "${datasets[${i}]}" \
        --model-type RandomForest \
        --dim "${dim}"
    done
done
