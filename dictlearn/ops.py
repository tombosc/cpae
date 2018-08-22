import numpy

import theano
from theano import tensor

from dictlearn.util import vec2str

class WordToIdOp(theano.Op):
    """Replaces words with their ids."""
    def __init__(self, vocab):
        self._vocab = vocab

    def make_node(self, input_):
        input_ = tensor.as_tensor_variable(input_)
        output_type = tensor.TensorType(
            input_.dtype, input_.broadcastable[:-1])
        return theano.Apply(self, [input_], [output_type()])

    def perform(self, node, inputs, output_storage):
        words = inputs[0]
        words_flat = words.reshape(-1, words.shape[-1])
        word_ids = numpy.array([self._vocab.word_to_id(vec2str(word))
                                for word in words_flat])
        output_storage[0][0] = word_ids.reshape(words.shape[:-1])

class WordToCountOp(theano.Op):
    """Replaces words with their counts."""
    def __init__(self, vocab):
        self._vocab = vocab

    def make_node(self, input_):
        input_ = tensor.as_tensor_variable(input_)
        output_type = tensor.TensorType(
            input_.dtype, input_.broadcastable[:-1])
        return theano.Apply(self, [input_], [output_type()])

    def perform(self, node, inputs, output_storage):
        words = inputs[0]
        words_flat = words.reshape(-1, words.shape[-1])
        word_counts = numpy.array([self._vocab.word_freq(vec2str(word))
                                for word in words_flat])
        output_storage[0][0] = word_counts.reshape(words.shape[:-1])

