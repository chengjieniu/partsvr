#!/bin/sh

python ../data_code/densify/densify.py \
--shapenet_path=dataset/ShapeNetCore.v1 \
--python_interpreter=python3 \
--synth_set=$1 \
--subset=test \
--output_dir=gt/dense
