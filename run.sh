#!/bin/bash

source activate tmp_cpae
export PYTHONPATH=$PWD:$PYTHONPATH
export FUEL_DATA_PATH=$PWD"/data/en_wn_full"
export THEANO_FLAGS="optimizer=fast_run,device=cuda0,floatX=float32,gpuarray.preallocate=1,allow_gc=False"

RESULTS_DIR="results/en_wn_full"
CONFIG_NAME="s2sg_w2v_defs_1_pen128"
#CONFIG_NAME="s2sg_enwiki_full_1_pen0"

mkdir -p $RESULTS_DIR

python bin/train_s2s.py --fast-start $CONFIG_NAME $RESULTS_DIR/$CONFIG_NAME
