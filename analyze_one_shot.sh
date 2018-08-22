#!/bin/sh

export PYTHONPATH=$PWD:$PYTHONPATH
export FUEL_DATA_PATH=$PWD"/data/en_wn_split"
# use the path of the embedding file of the model selected by the model selection procedure.
EMBEDDING_FN="?.pkl"
EMBEDDING_NAME="?"
DATA_DIR="data/en_wn_split/"
RESULTS_DIR="results/one_shot_analysis"

mkdir -p $RESULTS_DIR
python bin/eval_embs.py $EMBEDDING_FN dict_poly $DATA_DIR > $RESULTS_DIR/$EMBEDDING_NAME
