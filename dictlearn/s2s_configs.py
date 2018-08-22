from dictlearn.config_registry import ConfigRegistry

configs_ae = ConfigRegistry() 

configs_ae.set_root_config({
    # data_path: not useful to use that, it's better to use FUEL_DATA_PATH
    # so that we can keep identical configs for different dictionaries
    'data_path': '', 
    # the following param was useful to run a baseline without an encoder
    # would be similar to word2vec with only one target word (the defined word)
    # this is NOT the baseline in the paper, it is weaker than word2vec
    'vocab_keys_path': '',
    'layout' : 'dict', # don't change. TODO remove this option
    # num_input_words can be set lower than the number of lines in vocab.txt
    # this allows to replace rare words with UNK (for example, if set to all the words 
    # from line 10000 on will be replaced by UNK token if it is set to 10000)
    'num_input_words' : 10000,
    # same for num_output_words: the loss will ignore words that are ranked 
    # above the value
    'num_output_words': 10000,
    # max definition length
    'max_length' : 100,
    'batch_size' : 32,
    'batch_size_valid' : 32,

    # model
    'encoder': 'lstm', # experimental code with bilstm variants (see seq2seq.py)
    'decoder': 'skip-gram', # do not change?
    # You should use emb_dim = dim unless you're playing with more experimental
    # code.
    'emb_dim' : 300, 
    'dim' : 300,
    # Optimizer is adam.
    'learning_rate' : 0.0003,
    'momentum' : 0.9,
    'grad_clip_threshold' : 5.0,
    'shared_rnn': False, # when using a lstm encoder and decoder only
    # the translate layer is an optional linear layer that transforms
    # the last hidden state of the encoder to be the definition embedding
    'translate_layer': 'linear', 
    'word_dropout': 0.0, # probability of replacing a word with UNK in defs
    'tied_in_out': False, 
    'reconstruction_coef': 1, # You can set that to 0 to retrieve Hill's model

    # Load pretrained encoder embeddings
    # it's one of the 2 files (.txt) produced by "pack_glove_update_vocab.py"
    'vocab_path': "", 
    # it's one of the 2 files (.npy) produced by "pack_glove_update_vocab.py"
    'embedding_path': '',
    'freeze_pretrained': False, # set to True when using pretrained embeddings
    'provide_targets': False, # TODO remove from code

    # Consistency penalty
    'proximity_coef': 1, # the lambda coefficient for consistency penalty term
    'proximity_distance': 'l2', # distance: 'l1', 'l2', or 'cos'

    # monitoring and checkpointing
    # the frequency in terms of batch for monitoring train/valid losses
    'mon_freq_train' : 200,
    # in the "full dictionary" setting, there is no validation
    'mon_freq_valid' : 0, # no validation
    'n_valid_early': 0, # no validation
    'save_freq_batches' : 0,
    'checkpoint_every_n_batches': 0,
    'checkpoint_every_n_epochs': 5,
    'n_epochs' : 50,
    'monitor_parameters' : False,
    'fast_checkpoint' : True,
    'seed': 1


})

c = configs_ae['root']
c['freeze_pretrained'] = True
c['emb_dim'] = 400
c['dim'] = 400
c['embedding_path'] = 'enwiki_full_archive.npy'
c['vocab_path'] = 'enwiki_full_vocab.txt'
c['num_input_words'] = 67861 # all input words
# ignore words w/ less than 5 counts in the output: speeds up training
c['num_output_words'] = 18408 
c['proximity_coef'] = 1
configs_ae['s2sg_enwiki_full_1_pen1'] = c
c['proximity_coef'] = 2
configs_ae['s2sg_enwiki_full_1_pen2'] = c
c['proximity_coef'] = 4
configs_ae['s2sg_enwiki_full_1_pen4'] = c
c['proximity_coef'] = 8
configs_ae['s2sg_enwiki_full_1_pen8'] = c
c['proximity_coef'] = 16
configs_ae['s2sg_enwiki_full_1_pen16'] = c
c['proximity_coef'] = 32
configs_ae['s2sg_enwiki_full_1_pen32'] = c
c['proximity_coef'] = 64
configs_ae['s2sg_enwiki_full_1_pen64'] = c
c['proximity_coef'] = 0
configs_ae['s2sg_enwiki_full_1_pen0'] = c


c['proximity_coef'] = 100
c['reconstruction_coef'] = 0
configs_ae['s2sg_enwiki_full_1_hill'] = c

c = configs_ae['root']
c['freeze_pretrained'] = True
c['emb_dim'] = 300
c['dim'] = 300
c['embedding_path'] = 'definition_SG_archive.npy'
c['vocab_path'] = 'definition_SG_vocab.txt'
c['num_input_words'] = 45102 # all input words
# ignore words w/ less than 5 counts in the output: speeds up training
c['num_output_words'] = 17546 
c['proximity_coef'] = 1
configs_ae['s2sg_w2v_defs_1_pen1'] = c
c['proximity_coef'] = 2
configs_ae['s2sg_w2v_defs_1_pen2'] = c
c['proximity_coef'] = 4
configs_ae['s2sg_w2v_defs_1_pen4'] = c
c['proximity_coef'] = 8
configs_ae['s2sg_w2v_defs_1_pen8'] = c
c['proximity_coef'] = 16
configs_ae['s2sg_w2v_defs_1_pen16'] = c
c['proximity_coef'] = 32
configs_ae['s2sg_w2v_defs_1_pen32'] = c
c['proximity_coef'] = 64
configs_ae['s2sg_w2v_defs_1_pen64'] = c
c['proximity_coef'] = 0
configs_ae['s2sg_w2v_defs_1_pen0'] = c
# We set the proximity coef to 100. Should be equivalent to 1 (there is 
# gradient clipping) but maybe it is faster to optimise.
c['proximity_coef'] = 100
c['reconstruction_coef'] = 0
configs_ae['s2sg_w2v_defs_1_hill'] = c

c = configs_ae['root']
c['reconstruction_coef'] = 1
c['mon_freq_valid'] = 0
c['n_valid_early'] = 0
c['n_epochs'] = 50
c['num_input_words'] = 50000 # if more than the total number of words, as is 
# the case here, just reduces to the total number of words in the vocab file
# Different kind of checkpointing here
# It only saves one model, that's saved every 10000 batches.
c['save_freq_batches'] = 10000
c['checkpoint_every_n_epochs'] = 0
c['proximity_coef'] = 1
lm_config_registry['s2sg_c300_pen1_c1_F'] = c
c['proximity_coef'] = 2
lm_config_registry['s2sg_c300_pen2_c1_F'] = c
c['proximity_coef'] = 3
lm_config_registry['s2sg_c300_pen3_c1_F'] = c
c['proximity_coef'] = 4
lm_config_registry['s2sg_c300_pen4_c1_F'] = c
c['proximity_coef'] = 8
lm_config_registry['s2sg_c300_pen8_c1_F'] = c
c['proximity_coef'] = 16
lm_config_registry['s2sg_c300_pen16_c1_F'] = c
c['proximity_coef'] = 32
lm_config_registry['s2sg_c300_pen32_c1_F'] = c
c['proximity_coef'] = 64
lm_config_registry['s2sg_c300_pen64_c1_F'] = c
