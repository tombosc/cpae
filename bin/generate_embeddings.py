#!/usr/bin/env python
import os
from dictlearn.def_autoencoder_training import (initialize_data_and_model,
                                use_n_identical_keys, use_keys)
from dictlearn.s2s_configs import configs_ae
from dictlearn.main import main_generate_embeddings
from dictlearn.vocab import Vocabulary
from blocks.serialization import load_parameters
from blocks.model import Model
from blocks.filter import VariableFilter
from collections import Counter, defaultdict
import numpy as np
from dictlearn.util import vec2str, serialize_embeddings

import json
import theano
import theano.tensor as T

def generate_embeddings(config, tar_path, part, dest_path, format_, 
                        average = False, encoder_embeddings=None,
                        **kwargs):
    """
    generate embeddings for all the defintions, average them and serialize OR
    if encoder_embeddings, serialize the models' encoder embeddings

    config: name of the config of the model
    tar_path: tar path of the model parameters
    part: part of the dataset (should be either 'train', 'valid', 'test' or 'all')
    dest_path: directory where the serialized embeddings will be written
    format: either 'dict' or 'glove'
    encoder_embeddings: None, 'only', 'mixed', 'if_missing'
      - None: don't include encoder embeddings
      - 'only': don't read any data, just serialize the encoder embeddings
      - 'mixed': add the encoder embeddings to the list of definition embeddings
      - 'if_missing': add the encoder embeddings when there is no corresponding def
    average: if true, multi-prototype embeddings will be averaged
    """
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    c = config
    data, model = initialize_data_and_model(c, train_phase=False)
    words = T.ltensor3('words')
    words_mask = T.matrix('words_mask')
    keys = T.lmatrix('keys')
    n_identical_keys = T.lvector('n_identical_keys')
    sym_args = [words, words_mask]
        
    if format_ not in ['dict', 'glove']:
        raise ValueError("format should be either: dict, glove")

    if not c['encoder'] and encoder_embeddings != 'only':
        raise ValueError('Error: this model does not have an encoder.')

    if use_keys(c):
        sym_args.append(keys)
    if use_n_identical_keys(c):
        sym_args.append(n_identical_keys)

    costs = model.apply(*sym_args, train_phase=False)

    cg = Model(costs)

    with open(tar_path) as src:
        cg.set_parameter_values(load_parameters(src))

    if encoder_embeddings:
        if encoder_embeddings == 'only' and not c['encoder']:
            embeddings_array = model.get_def_embeddings_params('key').eval()
        else:
            embeddings_array = model.get_def_embeddings_params('main').eval()
        entries = model.get_embeddings_entries()
        enc_embeddings = {e: np.asarray(a) for e, a in zip(entries, embeddings_array)}
        if encoder_embeddings == 'only':
            serialize_embeddings(enc_embeddings, format_, dest_path,
                                "encoder_embeddings")
            return 0

    embeddings_var, = VariableFilter(name='embeddings')(cg)
    compute = dict({"embeddings": embeddings_var})
    if c['proximity_coef'] != 0:
        prox_var, = VariableFilter(name='proximity_term')(cg)
        compute["proximity_term"] = prox_var
    print "sym args", sym_args
    predict_f = theano.function(sym_args, compute)
    batch_size = 256 # size of test_unseen
    stream = data.get_stream(part, batch_size=batch_size, max_length=c['max_length'],
                             remove_keys = False, remove_n_identical_keys=False)
    raw_data = [] # list of dicts containing the inputs and computed outputs
    i=0
    vocab = model._vocab
    print "start computing"
    embeddings = defaultdict(list)
    for input_data in stream.get_epoch_iterator(as_dict=True):
        if i%10==0:
            print "iteration:", i
        words = input_data['words']
        words_mask = input_data['words_mask']
        keys = input_data['keys']
        n_identical_keys = input_data['n_identical_keys']
        args = [words, words_mask]
        if use_keys(c):
            args.append(keys)
        if use_n_identical_keys(c):
            args.append(n_identical_keys)

        to_save = predict_f(*args)
        for k, h in zip(keys, to_save['embeddings']):
            key = vec2str(k)
            if encoder_embeddings == 'if_missing':
                try:
                    del enc_embeddings[key]
                except KeyError:
                    pass
            embeddings[key].append(h)
        i+=1 

    if encoder_embeddings in ['mixed', 'if_missing']:
        for k, e in enc_embeddings.iteritems():
            embeddings[k].append(e)

    if encoder_embeddings=='mixed':
        prefix_fname='mix_e_'
    elif encoder_embeddings=='if_missing':
        prefix_fname='if_mis_e_'
    else:
        prefix_fname=''

    # combine:
    if average:
        mean_embeddings = {}
        for k in embeddings.keys():
            mean_embeddings[k] = np.mean(np.asarray(embeddings[k]), axis=0)
        serialize_embeddings(mean_embeddings, format_, dest_path, prefix_fname+"mean_embeddings")
    else:
        serialize_embeddings(embeddings, format_, dest_path, prefix_fname+"embeddings")
     

if __name__ == "__main__":
    print "generate embeddings"
    main_generate_embeddings(configs_ae, generate_embeddings)
