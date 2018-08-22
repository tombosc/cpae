#!/bin/sh

CUR_DIR=$PWD
export PYTHONPATH=$PWD:$PYTHONPATH
ROOT_DICT="data/en_wn_full"
CRAWLED_DICT="data/dict_wn.json"

mkdir -p $ROOT_DICT

echo "Remove stopwords, multi-word expressions and recover case"
python bin/filter_dict.py --remove_stop_words --remove_mwe --retrieve_original_case $CRAWLED_DICT $ROOT_DICT/all_not_merged.json

echo "Merge definitions"
python bin/merge_definitions.py --seed 0 $ROOT_DICT/all_not_merged.json $ROOT_DICT/all.json

echo "Build vocabulary file"
python bin/build_vocab.py $ROOT_DICT/all.json $ROOT_DICT/vocab.txt

# create symlinks from train.json to all, and empty validation
cd $ROOT_DICT
ln -s all.json train.json
echo "{}" > valid.json
echo "{}" > test.json
cd $CUR_DIR
