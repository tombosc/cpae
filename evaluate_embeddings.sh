#/bin/bash

# Generate embeddings based on trained models
# Then, evaluate them

# Set some flags.
export PYTHONPATH=$PWD:$PWD/word-embeddings-benchmarks:$PYTHONPATH
export FUEL_DATA_PATH=$PWD"/data/en_wn_full"
export THEANO_FLAGS="optimizer=fast_run,device=cuda0,floatX=float32,gpuarray.preallocate=1,allow_gc=False"
export WEB_DATA="$PWD"

# RESULTS_DIR contains directories which each contains a file named
# like MODEL_FNAME (which contains the trained model parameters)
# For each of these models, we compute the embeddings then evaluate them
RESULTS_DIR="results/en_wn_full"
# The final result is stored in EMB_DIR with a subdirectory structure 
# mirroring that of RESULTS_DIR
EMB_DIR="embeddings/en_wn_full"
# The model name should be changed, depending on the number of epochs
# TODO Could automate that and pick the latest created file?
MODEL_FNAME="training_state.tar.after_batch_132250.tar"

shopt -s extglob

for MODEL_DIR in $RESULTS_DIR/*w2v*
do
	MODEL=$(basename $MODEL_DIR)
	echo $MODEL_DIR
	echo "$MODEL in $RESULTS_DIR/$MODEL/$MODEL_FNAME"
	mkdir -p $EMB_DIR/${MODEL} 

	# generate embeddings
	python bin/generate_embeddings.py "$MODEL" "$RESULTS_DIR"/"$MODEL"/"$MODEL_FNAME" --dest="$EMB_DIR"/"$MODEL" --part=all --format='dict' --encoder-embeddings="if_missing" 
	# evaluate embeddings
	python word-embeddings-benchmarks/scripts/evaluate_on_all.py -f "$EMB_DIR/$MODEL/if_mis_e_embeddings.pkl" -o "$EMB_DIR/${MODEL}/eval" -p dict 2>&1 | tee "$EMB_DIR/${MODEL}/coverage_and_eval"
done
