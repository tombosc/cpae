# Dependencies

See `requirements.txt`. Some packages such as [blocks](http://blocks.readthedocs.io/en/latest/setup.html) and [fuel](http://fuel.readthedocs.io/en/latest/setup.html) should be installed with pip using the github link to the projects:

- `pip install git+git://github.com/mila-udem/blocks.git@stable -r https://raw.githubusercontent.com/mila-udem/blocks/stable/requirements.txt`
- `pip install git+git://github.com/mila-udem/fuel.git@stable`

This code is heavily based on the [dict_based_learning repo](https://github.com/tombosc/dict_based_learning).

We directly include the files of several softwares that are slightly modified:

- [Word Embeddings Benchmark](https://github.com/kudkudak/word-embeddings-benchmarks) which we have prepackaged into the archive because it is a modified version which includes more datasets and also reads specific model files. 
- [Retrofitting](https://github.com/mfaruqui/retrofitting) which corrects a minor bug and adds more options.

We also include a the wordnet dictionary (definitions only) in `data/dict_wn.json` and the license that goes with it in `data/wordnet_LICENSE`.

# Prepare the data

2. Run `./build_split_dict.sh` to build the split dictionary. 
3. Run `./build_full_dict.sh` to build the full dictionary. 

## Pretrained embeddings
In order to use pretrained embeddings, you need `.npy` archives that will be loaded as input embeddings into the model and frozen (not trained). Additionally, you will need a custom vocabulary. For that purpose, you can modify and use two different scripts `build_pretrained_archive.sh` and `build_pretrained_w2v_defs.sh`. The first one include words that have definitions but that do not appear in definitions, while the second one does not.

Once you have the custom vocabulary, you can create configurations for the new models into `dictlearn/s2s_configs.py`. We give the configurations for the full dump experiment, the (very similar) dictionary data with word2vec pretrained archive and the full dictionary experiment without any pretraining.

# Train

See `run.sh` and the corresponding configuration names in `dictlearn/s2s_configs.py` for how to run one specific experiment.

# Generate and evaluate embeddings

Once your model is trained, you can use it to generate embeddings for all the words which have a definition. Use `evaluate_embeddings.sh` to generate and evalute embeddings. It is not fully automatic (requires the right `.tar` archive that contains the trained model), so please read it to make sure that the filenames are coherent with the number of epochs that you have, etc. The script generates the scores on dev and test sets. You can use the notebook in `notebooks/eval_embs.ipynb` which shows how to do model selection.

There is a distinct script to evaluate the one-shot learning abilities of model: see `analyze_one_shot.sh`.

# Comparing against the baselines

- Hill's model is recovered (with shared embeddings between the encoder and decoder and a L2 distance instead of cosine) when `c['proximity_coef'] = 0` for the configuration `c`. So you can use the same code as for AE and CPAE to run that model.
- To do retrofitting, you can look into `preparation_retrofitting/README.md`.
- To use dict2vec, please look at `preparation_dict2vec/README.md`.

# Misc

In order to export definitions that are word2vec readable (using the naive concatenation scheme described in the paper), you can use `bin/export_definitions.py`.
If you are looking for something that's not described, please look at the scripts in `bin/`, there might be something undocumented that can help you.
