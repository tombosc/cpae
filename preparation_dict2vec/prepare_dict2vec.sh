#!/bin/bash

DEFS_FN="../data/en_wn_full/all.json"
DEFS_FN_TXT="defs.txt"
EMBEDDING_ORIGINAL="../data/enwiki_50M.txt"
CLEANED_DEFS="defs_cleaned_50M.txt"
VOCAB_FILE=${EMBEDDING_ORIGINAL}_vocab.txt
STRONG_FN=strong_pairs_50M.txt
WEAK_FN=weak_pairs_50M.txt
K=5

echo "Format from json"
python format_from_json.py $DEFS_FN Wn $DEFS_FN_TXT
cat $EMBEDDING_ORIGINAL | cut -d' ' -f1 > $VOCAB_FILE

echo "Clean"
python clean_definitions.py -d $DEFS_FN_TXT -v $VOCAB_FILE 
# creates all-definitions-cleaned.txt
mv all-definitions-cleaned.txt $CLEANED_DEFS

echo "Generate pairs"
# then, use dict2vec's code to generate pairs with these parameters.
#python3 generate_pairs.py -d $CLEANED_DEFS -e $EMBEDDING_ORIGINAL -K $K -sf $STRONG_FN -wf $WEAK_FN
