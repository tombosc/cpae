import os
import time
import socket
import atexit
import signal
import pprint
import logging
import cPickle
import subprocess
import json

import numpy
import theano
from theano import tensor

import blocks
from blocks.initialization import Uniform, Constant
from blocks.algorithms import (
    Adam, GradientDescent, Adam, StepClipping, CompositeRule)
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.training import TrackTheBest
from blocks.extensions.saveload import Load, Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
                                          
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.predicates import OnLogRecord

from blocks.main_loop import MainLoop
from blocks.serialization import load_parameters

import fuel
from fuel.streams import ServerDataStream

from dictlearn.util import rename, masked_root_mean_square, get_free_port
from dictlearn.theano_util import parameter_stats
from dictlearn.data import LanguageModellingData
from dictlearn.extensions import (
    StartFuelServer, LoadNoUnpickling,
    IntermediateCheckpoint)
from dictlearn.retrieval import Retrieval, Dictionary
from dictlearn.seq2seq import Seq2Seq
from dictlearn.vocab import Vocabulary

logger = logging.getLogger()

def use_keys(c):
    return c['proximity_coef'] > 0 or not c['encoder'] or c['provide_targets']

def use_n_identical_keys(c):
    return False 

def initialize_data_and_model(config, train_phase, layout='dict'):
    c = config
    fuel_path = fuel.config.data_path[0]
    vocab_main = None
    vocab_keys = None
    if not c['encoder']:
        if not c['vocab_keys_path']:
            raise ValueError('Error: Should specify vocab_keys_path when no encoder')
        vocab_keys = Vocabulary(
            os.path.join(fuel.config.data_path[0], c['vocab_keys_path']))

        
    if c['vocab_path']:
        vocab_main = Vocabulary(
            os.path.join(fuel.config.data_path[0], c['vocab_path']))
    # TODO: change name of class LanguageModellingData... very ill-named.
    data = LanguageModellingData(c['data_path'], layout, vocab=vocab_main)

    vocab_main = data.vocab

    model = Seq2Seq(c['emb_dim'], c['dim'], c['num_input_words'],
                       c['num_output_words'], data.vocab,
                       proximity_coef = c['proximity_coef'],
                       proximity_distance = c['proximity_distance'],
                       encoder = c['encoder'],
                       decoder = c['decoder'],
                       shared_rnn = c['shared_rnn'],
                       translate_layer = c['translate_layer'],
                       word_dropout = c['word_dropout'],  
                       tied_in_out = c['tied_in_out'],
                       vocab_keys = vocab_keys,
                       reconstruction_coef = c['reconstruction_coef'],  
                       provide_targets = c['provide_targets'],
                       weights_init=Uniform(width=0.1),
                       biases_init=Constant(0.))
                       
    model.initialize()

    if c['embedding_path'] and ((train_phase or c['freeze_pretrained']) or
                                c['provide_targets']):
        if c['provide_targets'] and c['freeze_pretrained']:
            raise ValueError("Can't provide_targets and use freeze_pretrained."
                             "In that case, simply use freeze_pretrained")
                            
        # if encoder embeddings are frozen, then we should load them 
        # as they're not saved with the models parameters
        emb_full_path = os.path.join(fuel_path, c['embedding_path'])
        embedding_matrix = numpy.load(emb_full_path)
        if c['provide_targets']:
            model.set_def_embeddings(embedding_matrix, 'target')
            logger.debug("Pre-trained targets loaded")
        else:
            model.set_def_embeddings(embedding_matrix, 'main')
            logger.debug("Pre-trained encoder embeddings loaded")

    return data, model

def train_model(new_training_job, config, save_path, params,
                         fast_start, fuel_server, seed):
    c = config
    if seed:
        fuel.config.default_seed = seed
        blocks.config.config.default_seed = seed

    data, model = initialize_data_and_model(config, train_phase=True)

    # full main loop can be saved...
    main_loop_path = os.path.join(save_path, 'main_loop.tar')
    # or only state (log + params) which can be useful not to pickle embeddings
    state_path = os.path.join(save_path, 'training_state.tar')
    stream_path = os.path.join(save_path, 'stream.pkl')
    best_tar_path = os.path.join(save_path, "best_model.tar")

    keys = tensor.lmatrix('keys')
    n_identical_keys = tensor.lvector('n_identical_keys')
    words = tensor.ltensor3('words')
    words_mask = tensor.matrix('words_mask')
    if theano.config.compute_test_value != 'off':
        #TODO
        test_value_data = next(
            data.get_stream('train', batch_size=4, max_length=5)
            .get_epoch_iterator())
        words.tag.test_value = test_value_data[0]
        words_mask.tag.test_value = test_value_data[1]

    if use_keys(c) and use_n_identical_keys(c):
        costs = model.apply(words, words_mask, keys, n_identical_keys, train_phase=True)
    elif use_keys(c):
        costs = model.apply(words, words_mask, keys, train_phase=True)
    else:
        costs = model.apply(words, words_mask, train_phase=True)
    cost = rename(costs.mean(), 'mean_cost')

    cg = Model(cost)
    if params:
        logger.debug("Load parameters from {}".format(params))
        with open(params) as src:
            cg.set_parameter_values(load_parameters(src))

    length = rename(words.shape[1], 'length')
    perplexity, = VariableFilter(name='perplexity')(cg)
    monitored_vars = [length, cost, perplexity]
    if c['proximity_coef']:
        proximity_term, = VariableFilter(name='proximity_term')(cg)
        monitored_vars.append(proximity_term)

    print "inputs of the model:", cg.inputs

    parameters = cg.get_parameter_dict()
    trained_parameters = parameters.values()
    saved_parameters = parameters.values()
    if c['embedding_path']:
        if c['freeze_pretrained']:
            logger.debug("Exclude pretrained encoder embeddings from the trained parameters")
            to_freeze='main'
        elif c['provide_targets']:
            logger.debug("Exclude pretrained targets from the trained parameters")
            to_freeze='target'
        trained_parameters = [p for p in trained_parameters
                              if not p == model.get_def_embeddings_params(to_freeze)]
        saved_parameters = [p for p in saved_parameters
                              if not p == model.get_def_embeddings_params(to_freeze)]

    logger.info("Cost parameters" + "\n" +
                pprint.pformat(
                    [" ".join((
                       key, str(parameters[key].get_value().shape),
                       'trained' if parameters[key] in trained_parameters else 'frozen'))
                     for key in sorted(parameters.keys())],
                    width=120))

    rules = []
    if c['grad_clip_threshold']:
        rules.append(StepClipping(c['grad_clip_threshold']))
    rules.append(Adam(learning_rate=c['learning_rate'],
                      beta1=c['momentum']))
    algorithm = GradientDescent(
        cost=cost,
        parameters=trained_parameters,
        step_rule=CompositeRule(rules))

    train_monitored_vars = list(monitored_vars)
    if c['grad_clip_threshold']:
        train_monitored_vars.append(algorithm.total_gradient_norm)

    if c['monitor_parameters']:
        train_monitored_vars.extend(parameter_stats(parameters, algorithm))


    # We use a completely random seed on purpose. With Fuel server
    # it's currently not possible to restore the state of the training
    # stream. That's why it's probably better to just have it stateless.
    stream_seed = numpy.random.randint(0, 10000000) if fuel_server else None
    training_stream = data.get_stream(
        'train', batch_size=c['batch_size'], max_length=c['max_length'],
        seed=stream_seed, remove_keys=not use_keys(c),
        remove_n_identical_keys=not use_n_identical_keys(c))
    print "trainin_stream will contains sources:", training_stream.sources

    original_training_stream = training_stream
    if fuel_server:
        # the port will be configured by the StartFuelServer extension
        training_stream = ServerDataStream(
            sources=training_stream.sources,
            produces_examples=training_stream.produces_examples)

    validate = c['mon_freq_valid'] > 0

    if validate:
        valid_stream = data.get_stream('valid', batch_size=c['batch_size_valid'],
                                   max_length=c['max_length'], seed=stream_seed,
                                   remove_keys=not use_keys(c),
                                   remove_n_identical_keys=not use_n_identical_keys(c))
        validation = DataStreamMonitoring(
            monitored_vars,
            valid_stream,
            prefix="valid").set_conditions(
                before_first_epoch=not fast_start,
                on_resumption = True,
                every_n_batches=c['mon_freq_valid'])
        track_the_best = TrackTheBest(
                validation.record_name(cost),
                choose_best=min).set_conditions(
                on_resumption = True,
                after_epoch=True,
                every_n_batches=c['mon_freq_valid'])

    # don't save them the entire main loop to avoid pickling everything
    if c['fast_checkpoint']:
        cp_path = state_path
        load = (LoadNoUnpickling(cp_path, load_iteration_state=True, load_log=True)
            .set_conditions(before_training=not new_training_job))
        cp_args = {
            'save_main_loop': False,
            'save_separately' : ['log', 'iteration_state'],
            'parameters': saved_parameters
        }

    else:
        cp_path = main_loop_path
        load = (Load(cp_path, load_iteration_state=True, load_log=True)
            .set_conditions(before_training=not new_training_job))
        cp_args = {
            'save_separately' : ['iteration_state'],
            'parameters': saved_parameters
        }


    checkpoint = Checkpoint(cp_path,
                            before_training=not fast_start,
                            every_n_batches=c['save_freq_batches'],
                            after_training=not fast_start,
                            **cp_args)

    if c['checkpoint_every_n_batches'] > 0 or c['checkpoint_every_n_epochs'] > 0:
        intermediate_cp = IntermediateCheckpoint(
                             cp_path,
                             every_n_epochs=c['checkpoint_every_n_epochs'],
                             every_n_batches=c['checkpoint_every_n_batches'],
                             after_training=False,
                             **cp_args)

    if validate:
        checkpoint = checkpoint.add_condition(
                                ['after_batch', 'after_epoch'],
                                OnLogRecord(track_the_best.notification_name),
                                (best_tar_path,))

    extensions = [
            load,
            StartFuelServer(original_training_stream,
                            stream_path,
                            before_training=fuel_server),
            Timing(every_n_batches=c['mon_freq_train'])
        ]

    extensions.extend([
        TrainingDataMonitoring(
            train_monitored_vars, prefix="train",
            every_n_batches=c['mon_freq_train']),
        ])
    if validate:
        extensions.extend([validation, track_the_best])

    extensions.append(checkpoint)
    if c['checkpoint_every_n_batches'] > 0 or c['checkpoint_every_n_epochs'] > 0:
        extensions.append(intermediate_cp)
    extensions.extend([
        Printing(on_resumption=True,
                 every_n_batches=c['mon_freq_train'])
    ])

    if validate and c['n_valid_early'] > 0:
        extensions.append(
            FinishIfNoImprovementAfter(
                track_the_best.notification_name,
                iterations=c['n_valid_early'] * c['mon_freq_valid'],
                every_n_batches=c['mon_freq_valid'])
        )
    extensions.append(
        FinishAfter(after_n_epochs=c['n_epochs'])
    )

    logger.info("monitored variables during training:" + "\n" +
                pprint.pformat(train_monitored_vars, width=120))
    logger.info("monitored variables during valid:" + "\n" +
                pprint.pformat(monitored_vars, width=120))


    main_loop = MainLoop(
        algorithm,
        training_stream,
        model=Model(cost),
        extensions=extensions)

    main_loop.run()
