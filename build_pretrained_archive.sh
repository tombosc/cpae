#!/bin/sh

export PYTHONPATH=$PWD:$PYTHONPATH
ROOT_DICT="data/en_wn_full"
PRETRAINED_EMBEDDINGS="data/enwiki_full.txt"
PRETRAINED_NAME="enwiki_full"
# create pretrained embeddings

# we create a vocabulary that also contains the defined words. They need an 
# embedding too, even if they don't appear in the definition, so that the
# consistency penalty can be applied.
python bin/build_vocab.py $ROOT_DICT/all.json $ROOT_DICT/vocab_wkeys.txt --with-keys

python bin/pack_glove_update_vocab.py --try-lowercase $PRETRAINED_EMBEDDINGS $ROOT_DICT/${PRETRAINED_NAME}_archive $ROOT_DICT/${PRETRAINED_NAME}_vocab.txt --vocab $ROOT_DICT/vocab_wkeys.txt
