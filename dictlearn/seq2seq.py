"""A seq2seq model"""
import theano
import theano.tensor as T

from blocks.bricks import (Initializable, Linear, NDimensionalSoftmax, MLP,
                           Tanh, Rectifier, Logistic, Softmax)
from blocks.bricks.base import application
from blocks.bricks.recurrent import LSTM
from blocks.bricks.lookup import LookupTable
from blocks.initialization import Constant
from blocks.bricks.recurrent.misc import Bidirectional
from dictlearn.bidirectional_sum import BidirectionalSum
from blocks.roles import add_role, WEIGHT
from blocks.utils import shared_floatx_nans


from dictlearn.theano_util import get_dropout_mask
from dictlearn.ops import WordToIdOp, WordToCountOp
from dictlearn.aggregation_schemes import Perplexity
from dictlearn.util import masked_root_mean_square
from theano.tensor.shared_randomstreams import RandomStreams

floatX = theano.config.floatX


class Seq2Seq(Initializable):
    """ seq2seq model

    Parameters
    ----------
    emb_dim: int
        The dimension of word embeddings (including for def model if standalone)
    dim : int
        The dimension of the RNNs states (including for def model if standalone)
    num_input_words : int
        The size of the LM's input vocabulary.
    num_output_words : int
        The size of the LM's output vocabulary.
    vocab
        The vocabulary object.
    """
    def __init__(self, emb_dim, dim, num_input_words, 
                 num_output_words, vocab,
                 proximity_coef=0, proximity_distance='l2', encoder = 'lstm',
                 decoder = 'lstm', shared_rnn = False,
                 translate_layer = None, word_dropout= 0.,
                 tied_in_out = False, vocab_keys = None,
                 seed = 0, reconstruction_coef = 1.,
                 provide_targets = False,
                 **kwargs):
        """
        translate_layer: either a string containing the activation function to use
                         either a list containg the list of activations for a MLP
        """
        if emb_dim == 0:
            emb_dim = dim
        if num_input_words == 0:
            num_input_words = vocab.size()
        if num_output_words == 0:
            num_output_words = vocab.size()

        self._word_dropout = word_dropout

        self._tied_in_out = tied_in_out
       
        if not encoder:
            if proximity_coef:
                raise ValueError("Err: meaningless penalty term (no encoder)")
            if not vocab_keys:
                raise ValueError("Err: specify a key vocabulary (no encoder)")

        if tied_in_out and num_input_words != num_output_words:
            raise ValueError("Can't tie in and out embeddings. Different "
                             "vocabulary size")
        if shared_rnn and (encoder!='lstm' or decoder!='lstm'):
            raise ValueError("can't share RNN because either encoder or decoder"
                             "is not an RNN")
        if shared_rnn and decoder=='lstm_c':
            raise ValueError("can't share RNN because the decoder takes different"
                             "inputs")
        if word_dropout < 0 or word_dropout > 1:
            raise ValueError("invalid value for word dropout", str(word_dropout))
        if proximity_distance not in ['l1', 'l2', 'cos']:
             raise ValueError("unrecognized distance: {}".format(proximity_distance))

            
        if proximity_coef and emb_dim!=dim and not translate_layer:
            raise ValueError("""if proximity penalisation, emb_dim should equal dim or 
                              there should be a translate layer""")

        if encoder not in [None, 'lstm', 'bilstm', 'mean', 'weighted_mean', 'max_bilstm',
                           'bilstm_sum', 'max_bilstm_sum']:
            raise ValueError('encoder not recognized')
        if decoder not in ['skip-gram', 'lstm', 'lstm_c']:
            raise ValueError('decoder not recognized')

        self._proximity_distance = proximity_distance
        self._decoder = decoder
        self._encoder = encoder
        self._num_input_words = num_input_words
        self._num_output_words = num_output_words
        self._vocab = vocab
        self._proximity_coef = proximity_coef
        self._reconstruction_coef = reconstruction_coef
        self._provide_targets = provide_targets

        self._word_to_id = WordToIdOp(self._vocab)
        if vocab_keys:
            self._key_to_id = WordToIdOp(vocab_keys)

        children = []

        if encoder or (not encoder and decoder in ['lstm', 'lstm_c']):
            self._main_lookup = LookupTable(self._num_input_words, emb_dim, name='main_lookup')
            children.append(self._main_lookup)
        if provide_targets:
            # this is useful to simulate Hill's baseline without pretrained embeddings 
            # in the encoder, only as targets for the encoder.
            self._target_lookup = LookupTable(self._num_input_words, emb_dim, name='target_lookup')
            children.append(self._target_lookup)
        if not encoder:
            self._key_lookup = LookupTable(vocab_keys.size(), emb_dim, name='key_lookup')
            children.append(self._key_lookup)
        elif encoder == 'lstm':
            self._encoder_fork = Linear(emb_dim, 4 * dim, name='encoder_fork')
            self._encoder_rnn = LSTM(dim, name='encoder_rnn')
            children.extend([self._encoder_fork, self._encoder_rnn])
        elif encoder in ['bilstm', 'max_bilstm']:
            # dim is the dim of the concatenated vector
            self._encoder_fork = Linear(emb_dim, 2 * dim, name='encoder_fork')
            self._encoder_rnn = Bidirectional(LSTM(dim / 2, name='encoder_rnn'))
            children.extend([self._encoder_fork, self._encoder_rnn])
        elif encoder in ['bilstm_sum', 'max_bilstm_sum']:
            self._encoder_fork = Linear(emb_dim, 4 * dim, name='encoder_fork')
            self._encoder_rnn = BidirectionalSum(LSTM(dim, name='encoder_rnn'))
            children.extend([self._encoder_fork, self._encoder_rnn])
        elif encoder == 'mean':
            pass
        elif encoder == 'weighted_mean':
            self._encoder_w = MLP([Logistic()], [dim, 1], name="encoder_weights")
            children.extend([self._encoder_w])
        else:
            raise NotImplementedError()

        if decoder in ['lstm', 'lstm_c']:
            dim_after_translate = emb_dim
            if shared_rnn:
                self._decoder_fork = self._encoder_fork
                self._decoder_rnn = self._encoder_rnn
            else:
                if decoder=='lstm_c':
                    dim_2 = dim+emb_dim
                else:
                    dim_2 = dim
                self._decoder_fork = Linear(dim_2, 4 * dim, name='decoder_fork')
                self._decoder_rnn = LSTM(dim, name='decoder_rnn')
            children.extend([self._decoder_fork, self._decoder_rnn])
        elif decoder == 'skip-gram':
            dim_after_translate = emb_dim

        self._translate_layer = None
        activations = {'sigmoid': Logistic(), 
                       'tanh': Tanh(),
                       'linear': None}

        if translate_layer:
            if type(translate_layer) == str:
                translate_layer = [translate_layer]
            assert(type(translate_layer) == list)
            activations_translate = [activations[a] for a in translate_layer]
            dims_translate = [dim,]*len(translate_layer) + [dim_after_translate]
            self._translate_layer = MLP(activations_translate,
                                        dims_translate, name="translate_layer")
            children.append(self._translate_layer)

        if not self._tied_in_out:
            self._pre_softmax = Linear(emb_dim, self._num_output_words)
            children.append(self._pre_softmax)
        if decoder in ['lstm', 'lstm_c']:
            self._softmax = NDimensionalSoftmax()
        elif decoder in ['skip-gram']:
            self._softmax = Softmax()
        children.append(self._softmax)

        super(Seq2Seq, self).__init__(children=children, **kwargs)

    def _allocate(self):
        pass

    def _initialize(self):
        pass

    def get_embeddings_entries(self):
        return self._vocab.words

    def set_def_embeddings(self, embeddings, lookup='main'):
        if lookup == 'main':
            self._main_lookup.parameters[0].set_value(embeddings.astype(floatX))
        elif lookup == 'target':
            self._target_lookup.parameters[0].set_value(embeddings.astype(floatX))
        else:
            raise ValueError('Requested embedding not understood')

    def get_def_embeddings_params(self, lookup='main'):
        if lookup == 'main':
            return self._main_lookup.parameters[0]
        elif lookup == 'key':
            return self._key_lookup.parameters[0]
        elif lookup == 'target':
            return self._target_lookup.parameters[0]

        else:
            raise ValueError('Requested embedding not understood')

    def add_perplexity_measure(self, application_call, minus_logs, mask, name):
        sum_ce = (minus_logs * mask).sum()
        perplexity = T.exp(sum_ce / mask.sum())
        perplexity.tag.aggregation_scheme = Perplexity(
            sum_ce, mask.sum())
        application_call.add_auxiliary_variable(perplexity, name=name)
        return sum_ce / mask.sum()

    @application
    def apply(self, application_call, words, mask, keys=None,
              n_identical_keys=None, train_phase=True):
        """Compute the log-likelihood for a batch of sequences.

        words
            An integer matrix of shape (B, T), where T is the number of time
            step, B is the batch size. Note that this order of the axis is
            different from what all RNN bricks consume, hence and the axis
            should be transposed at some point.
        mask
            A float32 matrix of shape (B, T). Zeros indicate the padding.
        keys
            An integer matrix of shape (B). It contains the words that are 
            defined in the corresponding rows in words.

        """
        if not keys and self._proximity_coef != 0:
            raise ValueError("Err: should provide keys when using penalty term")

        if not self._encoder and not keys:
            raise ValueError("Err: should provide keys (no encoder)")

        word_ids = self._word_to_id(words)
        if keys:
            key_ids = self._word_to_id(keys)

        # dropout

        unk = self._vocab.unk
        if self._word_dropout > 0 and train_phase:
            dropout_mask = T.ones_like(word_ids, dtype=int)
            dropout_mask = get_dropout_mask(dropout_mask, self._word_dropout)
            # this gives a matrix of 0 (dropped word) and ones (kept words)
            # replace 0s by unk token and 1s by word ids
            word_ids_dropped = (T.eq(dropout_mask, 1) * word_ids +
                                T.eq(dropout_mask, 0) * unk)
            word_ids_in = word_ids_dropped
        else:
            word_ids_in = word_ids


        # shortlisting
        # input_word_ids uses word dropout

        input_word_ids = (T.lt(word_ids_in, self._num_input_words) * word_ids_in
                        + T.ge(word_ids_in, self._num_input_words) * unk)
        output_word_ids = (T.lt(word_ids, self._num_output_words) * word_ids
                          + T.ge(word_ids, self._num_output_words) * unk)

        if self._encoder or self._decoder != 'skip-gram':
            input_embeddings = self._main_lookup.apply(input_word_ids)


        # Encoder

        if self._encoder == 'lstm' or 'bilstm' in self._encoder:
            encoder_rnn_states = self._encoder_rnn.apply(
                T.transpose(self._encoder_fork.apply(input_embeddings), (1,0,2)),
                mask=mask.T)[0]
            
            if self._encoder in ['lstm', 'bilstm', 'bilstm_sum']:
                gen_embeddings = encoder_rnn_states[-1]
            elif self._encoder in ['max_bilstm', 'max_bilstm_sum']:
                mask_bc = T.addbroadcast(mask.dimshuffle(0,1,'x'),2) # (bs,L,dim)
                gen_embeddings = (input_embeddings * mask_bc + (1-mask_bc) * -10**8).max(axis=1)
            else:
                raise ValueError("encoder {} apply not specific".format(self._encoder))
        elif self._encoder == 'mean':
            mask_bc = T.addbroadcast(mask.dimshuffle(0,1,'x'),2)
            gen_embeddings = (input_embeddings * mask_bc).mean(axis=1)
        elif self._encoder == 'weighted_mean':
            mask_bc = T.addbroadcast(mask.dimshuffle(0,1,'x'),2)
            weights = self._encoder_w.apply(input_embeddings)
            weights = T.addbroadcast(weights, 2)
            weights = weights * mask_bc
            gen_embeddings = (input_embeddings * weights).mean(axis=1)
        elif not self._encoder:
            gen_embeddings = self._key_lookup.apply(key_ids)
        else:
            raise NotImplementedError()


        # Optional translation layer
        
        if self._translate_layer:
            in_decoder = self._translate_layer.apply(gen_embeddings)
        else:
            in_decoder = gen_embeddings # (bs, dim)
       
        application_call.add_auxiliary_variable(
                in_decoder.copy(), name="embeddings")

        # Decoder

        if self._decoder in ['lstm', 'lstm_c']:
            if self._decoder == 'lstm_c':
                tiled_in_decoder = T.tile(in_decoder.dimshuffle(0, 'x', 1), 
                                              (input_embeddings.shape[1], 1))
                input_embeddings = T.concatenate([input_embeddings,
                                                  tiled_in_decoder],
                                                 axis=2)
            
            decoded = self._decoder_rnn.apply(
                inputs=T.transpose(self._decoder_fork.apply(input_embeddings),
                                                            (1,0,2)),
                mask=mask.T,
                states=in_decoder)[0] # size (L, bs, dim)
            n_dim_decoded = 3
        elif self._decoder == 'skip-gram':
            decoded = in_decoder # size (bs, dim)
            n_dim_decoded = 2
        else:
            raise NotImplementedError()

        # we ignore the <bos> token
        targets = output_word_ids.T[1:] # (L-1, bs)
        targets_mask = mask.T[1:] # (L-1,bs)

        # Compute log probabilities

        if n_dim_decoded == 2:
            # Case where we have only one distrib for all timesteps: skip-gram
            if self._tied_in_out:
                W_out = self.get_def_embeddings_params().transpose() # (dim, V)
                logits = T.dot(decoded, W_out) # (bs, dim) x (dim,V) = (bs,V)
            else:
                logits = self._pre_softmax.apply(decoded) # (bs, V)
            size_batch, length_sentence = output_word_ids.shape
            normalized_logits = self._softmax.log_probabilities(logits) # (bs, V)
            indices = (targets.T + T.addbroadcast((T.arange(size_batch) * logits.shape[1]).dimshuffle(0, 'x'), 1)).flatten() # (bs*L)
            minus_logs = - normalized_logits.flatten()[indices].reshape((size_batch, length_sentence-1)).T # (L-1, bs)

        elif n_dim_decoded == 3:
            # Case where decoding is time dependent: recurrent decoders
            if self._tied_in_out:
                raise NotImplementedError()
                # TODO: implement... seems annoying because we need to replace
                # in the already implemented block code
            else:
                logits = self._pre_softmax.apply(decoded[:-1]) # (L-1, bs, V)
                minus_logs = self._softmax.categorical_cross_entropy(
                    targets, logits, extra_ndim=1)
              
        avg_CE = self.add_perplexity_measure(application_call, minus_logs,
                                             targets_mask, "perplexity")
        costs = self._reconstruction_coef * avg_CE

        if self._proximity_coef > 0:
            if not self._encoder:
                key_ids = self._key_to_id(keys)
            else:
                key_ids = self._word_to_id(keys)

            # shortlist: if we don't use all the input embeddings, we need to shortlist
            # so that there isn't a key error
            key_ids = (T.lt(key_ids, self._num_input_words) * key_ids
                      + T.ge(key_ids, self._num_input_words) * unk)

            if self._provide_targets:
                key_embeddings = self._target_lookup.apply(key_ids) #(bs, emb_dim)
            else:
                key_embeddings = self._main_lookup.apply(key_ids) #(bs, emb_dim)
            # don't penalize on UNK:
            mask = T.neq(key_ids, unk) * T.lt(key_ids, self._num_input_words)

            # average over dimension, and then manual averaging using the mask
            eps = T.constant(10**-6)

            if self._proximity_distance in ['l1', 'l2']:
                if self._proximity_distance == 'l1':
                    diff_embeddings = T.abs_(key_embeddings - in_decoder)
                else:
                    diff_embeddings = (key_embeddings - in_decoder)**2

                mask = mask.reshape((-1,1))
                sum_proximity_term = T.sum(T.mean(diff_embeddings * mask, axis=1))
                proximity_term = sum_proximity_term / (T.sum(mask) + eps)

            elif self._proximity_distance == 'cos':
                # numerator
                # TODO: debug
                mask = mask.reshape((-1,1)) # (bs, 1)
                masked_keys = key_embeddings*mask 
                masked_gen = in_decoder*mask
                dot_product_vector = T.sum(masked_keys*masked_gen, axis=1) #(bs)
                # denominator
                product_sqr_norms = T.sum((masked_keys)**2, axis=1) * T.sum((masked_gen)**2, axis=1)
                denominator = T.sqrt(product_sqr_norms + eps) #(bs)
                proximity_term = - T.sum(dot_product_vector / denominator) / (T.sum(mask) + eps)

            application_call.add_auxiliary_variable(
                proximity_term.copy(), name="proximity_term")
            costs = costs + self._proximity_coef * proximity_term

        return costs
