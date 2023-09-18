#!/bin/bash

cd ../app_training/

CHANNELS=$1
MODEL_TYPES=$2
NR_OF_EPOCHS=$3

for channel in ${CHANNELS[@]}; do
    for model_type in ${MODEL_TYPES[@]}; do

        echo "starting to process " $channel " with model type " $model_type

        docker-compose run --rm app_training \
            train.py \
            --channel=$channel \
            --model_type=$model_type \
            --nr_of_epochs=$NR_OF_EPOCHS
    done
done