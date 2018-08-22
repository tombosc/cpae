from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import logging
import numpy

from six import text_type, string_types

logger = logging.getLogger()

class Vocabulary(object):
    """Class that holds a vocabulary for the dataset."""
    BOS = '<bos>' # beginning-of-sequence
    EOS = '<eos>' # end-of-sequence
    BOD = '<bod>' # beginning-of-definition
    EOD = '<eod>' # end-of-definition
    UNK = '<unk>' # unknown token
    SPECIAL_TOKEN_MAP = {
        BOS: 'bos',
        EOS: 'eos',
        BOD: 'bod',
        EOD: 'eod',
        UNK: 'unk'
    }

    def __init__(self, path_or_data):
        """Initialize the vocabulary.

        path_or_data
            Either a list of words or the path to it.
        top_k
            If not `None`, only the first `top_k` entries will be left.
            Note, this does not include the special tokens.

        """
        if isinstance(path_or_data, string_types):
            words_and_freqs = []
            with open(path_or_data) as f:
                for line in f:
                    word, freq_str = line.strip().split()
                    word = word.decode('utf-8')
                    freq = int(freq_str)
                    words_and_freqs.append((word, freq))
        else:
            words_and_freqs = path_or_data

        self._id_to_word = []
        self._id_to_freq = []
        self._word_to_id = {}
        self.bos = self.eos = -1
        self.bod = self.eod = -1
        self.unk = -1

        for idx, (word_name, freq) in enumerate(words_and_freqs):
            token_attr = self.SPECIAL_TOKEN_MAP.get(word_name)
            if token_attr is not None:
                setattr(self, token_attr, idx)

            self._id_to_word.append(word_name)
            self._id_to_freq.append(freq)
            self._word_to_id[word_name] = idx

        if -1 in [getattr(self, attr)
                  for attr in self.SPECIAL_TOKEN_MAP.values()]:
            raise ValueError("special token not found in the vocabulary")

    def size(self):
        return len(self._id_to_word)

    @property
    def words(self):
        return self._id_to_word

    @property
    def frequencies(self):
        return self._id_to_freq

    def word_to_id(self, word, top_k=None):
        id_ = self._word_to_id.get(word)
        if id_ is not None and not top_k or id_ < top_k:
            return id_
        return self.unk

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def word_freq(self, word):
        if not word in self._word_to_id:
            return 0
        return self._id_to_freq[self._word_to_id[word]]

    def decode(self, cur_ids):
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence):
        word_ids = [self.word_to_id(cur_word) for cur_word in sentence]
        return numpy.array(word_ids, dtype=numpy.int64)

    @staticmethod
    def build(text, top_k=None, sort_by='frequency'):
        """
        sort_by is either 'frequency' or 'lexicographical'
        """
        # For now let's use a very stupid tokenization
        if isinstance(text, str):
            with open(text) as file_:
                def data():
                    for line in file_:
                        for word in line.strip().split():
                            yield word
                counter = Counter(data())
            logger.info("Data is read")
        else:
            counter = Counter(text)
            for word in list(counter.keys()):
                if ' ' in word:
                    logger.error("can't have tokens with spaces, skip {}".format(word.encode('utf8')))
                    del counter[word]
        # It was not immediately clear to me
        # if counter.most_common() selects consistenly among
        # the words with the same counts. Hence, let's just sort.
        if sort_by == 'frequency':
            sortf = lambda x: (-x[1], x[0])
        elif sort_by == 'lexicographical':
            sortf = lambda x: (x[0], x[1])
        else:
            raise Exception("sort not understood:", sort_by)
        words_and_freqs = sorted(counter.items(), key=sortf)
        logger.info("Words are sorted")
        if top_k:
            words_and_freqs  = words_and_freqs[:top_k]
        words_and_freqs = (
            [(Vocabulary.BOS, 0),
             (Vocabulary.EOS, 0),
             (Vocabulary.BOD, 0),
             (Vocabulary.EOD, 0),
             (Vocabulary.UNK, 0)]
            + words_and_freqs)

        return Vocabulary(words_and_freqs)

    def save(self, filename):
        with open(filename, 'w') as f:
            for word, freq in zip(self._id_to_word, self._id_to_freq):

                # Note: if this fails for you make sure that words read
                # and used by Vocabulary were utf-8 encoded prior to that
                if not isinstance(word, text_type):
                    word = text_type(word, "utf-8")

                print(word.encode('utf-8'), freq, file=f)
