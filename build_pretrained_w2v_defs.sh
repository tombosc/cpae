#!/bin/sh

export PYTHONPATH=$PWD:$PYTHONPATH
ROOT_DICT="data/en_wn_full"
PRETRAINED_EMBEDDINGS="data/definition_SG.txt"
PRETRAINED_NAME="definition_SG"
# create pretrained embeddings


# Note that here, we do not use a special vocabulary that contains all the defined words.
# This is because we do not want the defined words to have a target that was 
# computed by word2vec. We only want word2vec to have good representations for the words 
# in the definitions.

python bin/pack_glove_update_vocab.py --try-lowercase $PRETRAINED_EMBEDDINGS $ROOT_DICT/${PRETRAINED_NAME}_archive $ROOT_DICT/${PRETRAINED_NAME}_vocab.txt --vocab $ROOT_DICT/vocab.txt


