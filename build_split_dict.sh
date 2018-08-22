#!/bin/sh

CUR_DIR=$PWD
export PYTHONPATH=$PWD:$PYTHONPATH
ROOT_DICT="data/en_wn_split"
CRAWLED_DICT="data/dict_wn.json"
VALID_SIZE="2.5" # for early stopping

mkdir -p $ROOT_DICT

echo "remove stopwords"
# in this set of experiment, I did not retrieve the lowercase or remove MWE
# I suspect removing MWE does not change a lot as the MWE expressions are not put
# in the training set 
python bin/filter_dict.py --remove_stop_words $CRAWLED_DICT $ROOT_DICT/all.json

echo "split the dict"
# have to do that before merging, otherwise we lose the info on similar senses
python bin/split_dict_core.py --seed=1 $ROOT_DICT/all.json $ROOT_DICT/ $VALID_SIZE

for S in "train" "valid" "test" "all"
do
	python bin/merge_definitions.py --seed 0 $ROOT_DICT/$S.json $ROOT_DICT/$S.json
done

exit
# build vocabulary file based on TRAIN (we ignore validation here)
python bin/build_vocab.py $ROOT_DICT/train $ROOT_DICT/vocab.txt

# create symlinks from train.json to all, and empty validation
cd $ROOT_DICT
ln -s all.json train.json
echo "{}" > valid.json
echo "{}" > test.json
