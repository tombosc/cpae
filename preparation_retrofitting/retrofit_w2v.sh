#!/bin/sh

INPUT="../data/enwiki_full.txt"
DIM="400"
LEXICON="wn_en_full_defs.txt"
LEX_NAME="wn_en_full"
N_ITER="10"
PREFIX_OUT_DIR="embeddings/enwiki_full_400_wn"

for ALPHA in 0.25 0.5 1 2 4
do
	MODEL_NAME=$(basename `dirname $INPUT`)
	OUTPUT_DIR="${PREFIX_OUT_DIR}/${MODEL_NAME}_retrofit_lex_${LEX_NAME}_iter_${N_ITER}_alpha_${ALPHA}"
	OUTPUT="$OUTPUT_DIR/vectors.txt"
	mkdir -p $OUTPUT_DIR
	python retrofit.py --header-embeddings -i $INPUT -l $LEXICON -o $OUTPUT -n $N_ITER --alpha $ALPHA
	cp retrofit.py $OUTPUT_DIR/
done
