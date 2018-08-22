"""Dataset layout and data preparation.
"""

import os
import functools
import h5py
import numpy
import logging

logger = logging.getLogger()

import fuel
from fuel.transformers import (
    Mapping, Batch, Padding, AgnosticSourcewiseTransformer,
    FilterSources, Transformer, Flatten)
from fuel.schemes import (
    SequentialExampleScheme, IterationScheme, ConstantScheme, ShuffledExampleScheme)
from fuel.streams import DataStream
from fuel.datasets import H5PYDataset

from dictlearn.vocab import Vocabulary
from dictlearn.datasets import (
    TextDataset, PutTextTransfomer, DictDataset)
from dictlearn.util import str2vec

# We have to pad all the words to contain the same
# number of characters.
MAX_NUM_CHARACTERS = 100
def _str2vec(word):
    return str2vec(word, MAX_NUM_CHARACTERS)


def vectorize(words):
    """Replaces words with vectors."""
    return [_str2vec(word) for word in words]

def cut_if_too_long(n_max, list_words):
    return list_words[:n_max]

def word_to_singleton_list(word):
    return [word]

def listify(example):
    return tuple(list(source) for source in example)


def add_bos(bos, source_data):
    source_data = list(source_data)
    return [bos] + source_data


def add_eos(eos, source_data):
    source_data = list(source_data)
    source_data.append(eos)
    return source_data


class SourcewiseMapping(AgnosticSourcewiseTransformer):
    def __init__(self, data_stream, mapping, *args, **kwargs):
        kwargs.setdefault('which_sources', data_stream.sources)
        super(SourcewiseMapping, self).__init__(
            data_stream, data_stream.produces_examples, *args, **kwargs)
        self._mapping = mapping

    def transform_any_source(self, source_data, _):
        return self._mapping(source_data)


class RandomSpanScheme(IterationScheme):
    requests_examples = True

    def __init__(self, dataset_size, span_size, seed=None):
        self._dataset_size = dataset_size
        self._span_size = span_size
        if not seed:
            seed = fuel.config.default_seed
        self._rng = numpy.random.RandomState(seed)

    def get_request_iterator(self):
        # As for now this scheme produces an infinite stateless scheme,
        # it can itself play the role of an iterator. If we want to add
        # a state later, this trick will not cut it any more.
        return self

    def __iter__(self):
        return self

    def next(self):
        start = self._rng.randint(0, self._dataset_size - self._span_size)
        return slice(start, start + self._span_size)


class Data(object):
    """Builds the data stream for different parts of the data.

    TODO: refactor, only leave the caching logic.

    """
    def __init__(self, path, layout, vocab=None):
        self._path = os.path.join(fuel.config.data_path[0], path)
        self._layout = layout
        if not self._layout in ['dict', 'standard', 'dict_custom']:
            raise "layout {} is not supported".format(self._layout)

        self._vocab = vocab
        self._dataset_cache = {}

    @property
    def vocab(self):
        if not self._vocab:
            logger.debug("Loading default vocab")
            self._vocab = Vocabulary(
                os.path.join(self._path, "vocab.txt"))
        return self._vocab

    def get_dataset_path(self, part_or_fname):
        """
        if layout=='dict_custom' then provide the filename directly
        """
        if self._layout == 'standard':
            part_map = {'train': 'train.txt',
                        'valid': 'valid.txt',
                        'test': 'test.txt',
                        'test_unseen': 'test_unseen.txt'}
        elif self._layout == 'dict':
            part_map = {'train': 'train.json',
                        'valid' : 'valid.json',
                        'test': 'test.json',
                        'all': 'all.json'}
        elif self._layout == 'dict_custom':
            return os.path.join(self._path, part_or_fname)
        else:
            raise NotImplementedError('Not implemented layout ' + self._layout)
        return os.path.join(self._path, part_map[part_or_fname])

    def get_dataset(self, part, max_length=None):
        if not part in self._dataset_cache:
            part_path = self.get_dataset_path(part)
            if self._layout in ['dict', 'dict_custom']:
                self._dataset_cache[part] = DictDataset(part_path, max_length)
            else:
                self._dataset_cache[part] = TextDataset(part_path, max_length)
        return self._dataset_cache[part]

    def get_stream(self, *args, **kwargs):
        raise NotImplementedError()


class LanguageModellingData(Data):

    def get_stream(self, part, batch_size=None, max_length=None, seed=None,
                   remove_keys=False, add_bos_=True, remove_n_identical_keys=True):
        dataset = self.get_dataset(part, max_length)
        if self._layout == 'lambada' and part == 'train':
            stream = DataStream(
                dataset,
                iteration_scheme=RandomSpanScheme(
                    dataset.num_examples, max_length, seed))
            stream = Mapping(stream, listify)
        else:
            stream = dataset.get_example_stream()

        if add_bos_:
            stream = SourcewiseMapping(stream, functools.partial(add_bos, Vocabulary.BOS), which_sources=('words'))
        if max_length != None:
            stream = SourcewiseMapping(stream, functools.partial(cut_if_too_long, max_length), which_sources=('words'))
        stream = SourcewiseMapping(stream, vectorize, which_sources=('words'))
        stream = SourcewiseMapping(stream, word_to_singleton_list, which_sources=('keys'))
        stream = SourcewiseMapping(stream, vectorize, which_sources=('keys'))
        stream = Flatten(stream, which_sources=('keys'))


        if self._layout == 'dict':
            if remove_keys:
                stream = FilterSources(stream, [source for source in stream.sources
                                                if source != 'keys'])
            if remove_n_identical_keys:
                print "remove identical keys"
                stream = FilterSources(stream, [source for source in stream.sources
                                                if source != 'n_identical_keys'])
        if not batch_size:
            return stream

        stream = Batch(
            stream,
            iteration_scheme=ConstantScheme(batch_size))

        stream = Padding(stream, mask_sources=('words'))
        #stream = Flatten(stream, which_sources=('n_identical_keys'))

        #if self._layout == 'dict':
        #    stream = FilterSources(stream, [source for source in stream.sources
        #                                    if source != 'keys_mask'])
        #    stream = FilterSources(stream, [source for source in stream.sources
        #                                    if source != 'n_identical_keys_mask'])
        return stream


def digitize(vocab, source_data):
    return numpy.array([vocab.encode(words) for words in source_data])


def keep_text(example):
    return {'contexts_text': vectorize(example['contexts']),
            'questions_text': vectorize(example['questions'])}

