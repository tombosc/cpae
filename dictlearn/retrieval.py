"""Retrieve the dictionary definitions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
import shutil
import traceback
from collections import Counter, defaultdict

import numpy
import json
import nltk
from nltk.corpus import wordnet
from wordnik import swagger, WordApi, AccountApi

import fuel

from dictlearn.util import vec2str
from dictlearn.corenlp import StanfordCoreNLP

logger = logging.getLogger(__name__)

_MIN_REMAINING_CALLS = 100
_SAVE_EVERY_CALLS = 100

class Dictionary(object):
    """The dictionary of definitions.

    The native representation of the dictionary is a mapping from a word
    to a list of definitions, each of which is a sequence of words. All
    the words are stored as strings.

    """
    def __init__(self, path=None):

        if not path.endswith("json"):
            raise Exception("Please pass path ending in .json")

        self._data = defaultdict(list)
        self._meta_data = {}
        self._path = path
        self._meta_path = path.replace(".json", "_meta.json")
        #self._tmp_path = os.path.join(os.path.dirname(path),
        #                              self._path + '.tmp')
        #self._meta_tmp_path = os.path.join(os.path.dirname(path),
        #    self._meta_path + '.tmp')
        self._tmp_path = self._path + ".tmp"
        self._meta_tmp_path = self._meta_path + ".tmp"
        if self._path:
            if os.path.exists(self._path):
                self.load()
            else:
                logger.warning("No dict was loaded; ignore if you are"
                               " creating new one")

    def load(self):
        with open(self._path, 'r') as src:
            # can't just assign because self._data should keep being a
            # defaultdict
            self._data.update(json.load(src))
        if os.path.exists(self._meta_path):
            with open(self._meta_path, 'r') as src:
                self._meta_data = json.load(src)

    def save(self):
        logger.debug("saving...")
        with open(self._tmp_path, 'w') as dst:
            json.dump(self._data, dst, indent=2)
        shutil.move(self._tmp_path, self._path)
        logger.debug("saving meta...")
        with open(self._meta_tmp_path, 'w') as dst:
            json.dump(self._meta_data, dst, indent=2)
        shutil.move(self._meta_tmp_path, self._meta_path)
        logger.debug("saved")

    def _wait_until_quota_reset(self):
        while True:
            status = self._account_api.getApiTokenStatus()
            logger.debug("{} remaining calls".format(status.remainingCalls))
            if status.remainingCalls >= _MIN_REMAINING_CALLS:
                logger.debug("Wordnik quota was resetted")
                self._remaining_calls = status.remainingCalls
                return
            logger.debug("sleep until quota reset")
            time.sleep(60.)

    def add_identity_mapping(self, vocab):
        for word in vocab.words:
            self._data[word].append([word])
            self._meta_data[word] = {"sourceDictionary": "identity"}
        self.save()

    def remove_out_of_vocabulary(self, vocab):
        """
        remove definitions that are outside of a vocabulary vocab
        """
        # TODO: remove from meta too
        # can't just zip because if meta is empty then nothing happens
        print("vocab size : {}".format(vocab.size()))
        print("dict len : {}".format(len(self._data)))
        count_del = 0
        for word in self._data.keys():
            word_id = vocab.word_to_id(word)
            if word_id == vocab.unk:
                del self._data[word]
                count_del += 1
                # del self._meta_data[word] # TODO:
        print("have deleted {} definitions".format(count_del))
        self.save()


    def add_spelling(self, vocab, only_if_no_def=True):
        for word in vocab.words:
            # only add spelling to the words without defs
            if only_if_no_def and self._data[word]:
                continue
            def_ = word
            if len(def_) > 10:
                def_ = u"{}-{}".format(def_[:5], def_[-5:])
            # to avoid overlapping of the vocabularies, let's add a #
            # before each character
            self._data[word].append(map(lambda char: u'#' + char, def_))
            self._meta_data[word] = {"sourceDictionary": "spelling"}
        self.save()


    def add_from_lowercase_definitions(self, vocab):
        """Add definitions of lowercase word to each word (concat)
        """
        added = 0
        no_def = 0
        for word in vocab.words:
            word_lower = word.lower()
            if word != word_lower:
                lower_defs = self._data.get(word_lower)
                # This can be quite slow. But this code will not be used
                # very often.
                if not lower_defs:
                    if lower_defs is None:
                        # This can happen when API just dies (then vocab has, dict doesnt)
                        logger.error("None def for " + word)
                        continue
                    no_def += 1
                    logger.warning("No defs for " + str(word_lower) + "," + str(word))
                else:
                    # Note: often empty, like Zeus -> zeus
                    for def_ in lower_defs:
                        if not def_ in self._data[word]:
                            added += 1
                            self._data[word].append(def_)

        logger.info("No def for {}".format(no_def))
        logger.info("Added {} new defs in add_from_lowercase_definitions".format(added))
        self.save()


    def add_from_lemma_definitions(self, vocab, try_lower=False):
        """Add lemma definitions for non-lemmas.

        This code covers the following scenario: supposed a dictionary is crawled,
        but only for word lemmas.

        """
        lemmatizer = nltk.WordNetLemmatizer()
        added = 0
        for word in vocab.words:
            word_list = [word, word.lower()] if try_lower else [word]

            for word_to_lemma in word_list:
                try:
                    for part_of_speech in ['a', 's', 'r', 'n', 'v']:
                        lemma = lemmatizer.lemmatize(word_to_lemma, part_of_speech)
                        lemma_defs = self._data.get(lemma)
                        if lemma != word and lemma_defs:
                            # This can be quite slow. But this code will not be used
                            # very often.
                            for def_ in lemma_defs:
                                if not def_ in self._data[word]:
                                    added += 1
                                    self._data[word].append(def_)
                except:
                    logger.error("lemmatizer crashed on {}".format(word))
                    logger.error(traceback.format_exc())
        logger.info("Added {} new defs in add_from_lemma_definitions".format(added))
        self.save()

    def add_dictname_to_defs(self, vocab):
        """Add dict name in front of each def"""
        raise NotImplementedError()

    def crawl_lemmas(self, vocab):
        """Add Wordnet lemmas as definitions."""
        lemmatizer = nltk.WordNetLemmatizer()
        for word in vocab.words:
            definitions = []
            try:
                for part_of_speech in ['a', 's', 'r', 'n', 'v']:
                    lemma = lemmatizer.lemmatize(word, part_of_speech)
                    if lemma != word and not [lemma] in definitions:
                        definitions.append([lemma])
            except:
                logger.error("lemmatizer crashed on {}".format(word))
            if definitions:
                self._data[word] = definitions
        self.save()

    def crawl_lowercase(self, vocab):
        """Add Wordnet lemmas as definitions."""
        for word in vocab.words:
            self._data[word] = [[word.lower()]]
            self._meta_data[word.lower()] = {"sourceDictionary": "lowercase"}
        self.save()

    def crawl_wordnet(self, corenlp_url):
        corenlp = StanfordCoreNLP(corenlp_url)
        for i, word in enumerate(wordnet.words()):
            if word in self._data:
                logger.info('skip a known word {}'.format(word))
                continue
            self._data[word] = []
            for synset in wordnet.synsets(word):
                def_ = corenlp.tokenize(synset.definition())[0]
                self._data[word].append(def_)
            if i % 10000 == 0:
                self.save()
        self.save()

    def crawl_wordnik(self, vocab, api_key, corenlp_url,
                      call_quota=15000, crawl_also_lowercase=False, crawl_also_lemma=False):

        """Download and preprocess definitions from Wordnik.

        vocab
            Vocabulary for which the definitions should be found.
        api_key
            The API key to use in communications with Wordnik.
        call_quota
            Maximum number of calls per hour.
        crawl_also_lowercase
            If true will add lowercase version of each word to crawl list
        crawl_also_lemma
            If true will also crawl lemma versions of words
            WARNING: Lemma of Cat is Cat! So if you want to have definition of "cat"
            you have to also pass crawl_also_lowercase!

        """
        corenlp = StanfordCoreNLP(corenlp_url)

        self._remaining_calls = call_quota
        self._last_saved = 0

        client = swagger.ApiClient(
            api_key, 'https://api.wordnik.com/v4')
        self._word_api = WordApi.WordApi(client)
        self._account_api = AccountApi.AccountApi(client)

        words = list(vocab.words)

        if crawl_also_lowercase:
            words_set = set(words)  # For efficiency

            logger.info("Adding lowercase words to crawl")
            lowercased = []
            for w in words:
                if w.lower() not in words_set:
                    lowercased.append(w.lower())
            logger.info("Crawling additional {} words".format(len(lowercased)))
            words.extend(sorted(lowercased))

        if crawl_also_lemma:
            words_set = set(words) # For efficiency

            logger.info("Adding lemmatized vrsions to crawl")
            lemmas = []
            original = []
            lemmatizer = nltk.WordNetLemmatizer()
            for w in words:
                if isinstance(w, str):
                    w = w.decode('utf-8')

                for part_of_speech in ['a', 's', 'r', 'n', 'v']:
                    lemma = lemmatizer.lemmatize(w, part_of_speech)
                    if lemma not in words_set:
                        lemmas.append(lemma)
                        original.append(w)
            logger.info("Crawling additional {} words".format(len(lemmas)))
            for id in numpy.random.choice(len(lemmas), 100):
                logger.info("Example:" + lemmas[id] + "," + original[id])
                words.extend(sorted(lemmas))

        # Here, for now, we don't do any stemming or lemmatization.
        # Stemming is useless because the dictionary is not indexed with
        # lemmas, not stems. Lemmatizers, on the other hand, can not be
        # fully trusted when it comes to unknown words.
        for word in words:
            if isinstance(word, str):
                word = word.decode('utf-8')

            if word in self._data:
                logger.debug(u"a known word {}, skip".format(word))
                continue

            if self._last_saved >= _SAVE_EVERY_CALLS:
                self.save()
                self._last_saved = 0

            # 100 is a safery margin, I don't want to DDoS Wordnik :)
            if self._remaining_calls < _MIN_REMAINING_CALLS:
                self._wait_until_quota_reset()
            try:
                if isinstance(word, str):
                    word = word.decode('utf-8')
                definitions = self._word_api.getDefinitions(word)
            except Exception:
                logger.error(u"error during fetching '{}'".format(word))
                logger.error(traceback.format_exc())
                continue
            self._remaining_calls -= 1
            self._last_saved += 1

            if not definitions:
                definitions = []
            self._data[word] = []
            for def_ in definitions:
                try:
                    # seems like definition text can be both str and unicode
                    text = def_.text
                    if def_.text is None:
                        continue
                    if isinstance(text, str):
                        text = text.decode('utf-8')
                    tokenized_def = corenlp.tokenize(text)[0]
                    self._data[word].append(tokenized_def)
                    self._meta_data[" ".join(tokenized_def)] = {"sourceDictionary": def_.sourceDictionary}
                except Exception:
                    logger.error("error during tokenizing '{}'".format(text))
                    logger.error(traceback.format_exc())
            logger.debug(u"definitions for '{}' fetched {} remaining".format(word, self._remaining_calls))
        self.save()
        self._last_saved = 0

    def num_entries(self):
        return len(self._data)

    def get_definitions(self, key):
        return self._data.get(key, [])


class Retrieval(object):

    def __init__(self, vocab_text, dictionary,
                 max_def_length=1000, with_too_long_defs='drop',
                 max_def_per_word=1000000, with_too_many_defs='random',
                 exclude_top_k=None, vocab_def=None,
                 add_bod_eod=True, seed=777):
        """Retrieves the definitions.
        vocab_text
            The vocabulary for text
        vocab_def
            The vocabulary for definitions
        dictionary
            The dictionary of the definitions.
        max_def_length
            Disregard definitions that are longer than that.
        exclude_top_k
            Do not provide defitions for the first top k
            words of the vocabulary (typically the most frequent ones).
        max_def_per_word
            Pick at most max_n_def definitions for each word
        """
        self._vocab_text = vocab_text
        self._add_bod_eod = add_bod_eod
        self._rng = numpy.random.RandomState(seed)
        if vocab_def is None:
            self._vocab_def = self._vocab_text
        else:
            self._vocab_def = vocab_def
        self._dictionary = dictionary
        if exclude_top_k == -1:
            logger.debug("Exclude definition of all dictionary words")
            exclude_top_k = vocab_text.size()
        self._exclude_top_k = exclude_top_k

        if all(numpy.array(self._vocab_text._id_to_freq) == 1) and exclude_top_k > 0:
            # Also note that after merging doing exclude_top_k on freqs in merged def/text is perhaps
            # confusing
            raise Exception("Cannot perform exclude_top_k based on vocabulary without frequency information.")

        self._max_def_length = max_def_length
        if with_too_long_defs not in {"drop", "crop"}:
            raise NotImplementedError("Not implemented " + with_too_long_defs)
        self._with_too_long_defs = with_too_long_defs

        self._max_def_per_word = max_def_per_word
        if with_too_many_defs not in {"random", "exclude"}:
            raise NotImplementedError("Not implemented " + with_too_many_defs)
        self._with_too_many_defs = with_too_many_defs

        self._debug_info = {
            "missed_word_sample": [],

            "N_words": 0,
            "N_excluded_words": 0,

            "N_distinct_words": 0,
            "N_missed_distinct_words": 0,

            "N_def": 0,
            "N_dropped_def": 0,

            "N_queried_words": 0,
            "N_missed_words": 0,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_rng" not in self.__dict__:
            self._rng = numpy.random.RandomState(777)

    def retrieve(self, batch):
        """Retrieves all definitions for a batch of words sequences.

        TODO: definitions of phrases, phrasal verbs, etc.

        Returns
        -------
        defs
            A list of word definitions, each definition is a list of words.
        def_map
            A list of triples (batch_index, time_step, def_index). Maps
            words to their respective definitions from `defs`.

        """
        definitions = []
        def_map = []
        word_def_indices = {}

        for seq_pos, sequence in enumerate(batch):
            for word_pos, word in enumerate(sequence):
                if isinstance(word, numpy.ndarray):
                    word = vec2str(word)
                if not word:
                    continue
                self._debug_info['N_words'] += 1
                word_id = self._vocab_text.word_to_id(word)
                if (self._exclude_top_k
                        and word_id != self._vocab_text.unk
                        and word_id < self._exclude_top_k):
                    self._debug_info['N_excluded_words'] += 1
                    continue

                if word not in word_def_indices:
                    word_def_indices[word] = []
                    # The first time a word is encountered in a batch
                    word_defs = self._dictionary.get_definitions(word)

                    if self._max_def_per_word < len(word_defs):
                        if self._with_too_many_defs == 'random':
                            word_defs = self._rng.choice(
                                word_defs, self._max_def_per_word, replace=False)
                        else:
                            word_defs = []

                    # Debug info
                    self._debug_info['N_distinct_words'] += 1
                    self._debug_info['N_missed_distinct_words'] += (len(word_defs) == 0)
                    # End of debug info

                    for i, def_ in enumerate(word_defs):
                        self._debug_info['N_def'] += 1

                        if  self._with_too_long_defs == 'drop':
                            if len(def_) > self._max_def_length:
                                self._debug_info['N_dropped_def'] += 1
                                continue
                        elif self._with_too_long_defs == 'crop':
                            def_ = def_[0:self._max_def_length]
                        else:
                            raise NotImplementedError()

                        final_def_ = []
                        if self._add_bod_eod:
                            final_def_.append(self._vocab_def.bod)
                        for token in def_:
                            final_def_.append(self._vocab_def.word_to_id(token))
                        if self._add_bod_eod:
                            final_def_.append(self._vocab_def.eod)
                        word_def_indices[word].append(len(definitions))
                        definitions.append(final_def_)

                # Debug info
                self._debug_info['N_queried_words'] += 1
                if len(word_def_indices[word]) == 0:
                    self._debug_info['N_missed_words'] += 1
                    if len(self._debug_info['missed_word_sample']) == 10000:
                        self._debug_info['missed_word_sample'][numpy.random.randint(10000)] = word
                    else:
                        self._debug_info['missed_word_sample'].append(word)
                # End of debug info

                for def_index in word_def_indices[word]:
                    def_map.append((seq_pos, word_pos, def_index))

        return definitions, def_map

    def retrieve_and_pad(self, batch):
        defs, def_map = self.retrieve(batch)
        if not defs:
            defs.append(self.sentinel_definition())
        # `defs` have variable length and have to be padded
        max_def_length = max(map(len, defs))
        def_array = numpy.zeros((len(defs), max_def_length), dtype='int64')
        def_mask = numpy.ones_like(def_array, dtype=fuel.config.floatX)
        for i, def_ in enumerate(defs):
            def_array[i, :len(def_)] = def_
            def_mask[i, len(def_):] = 0.
        def_map = (numpy.array(def_map)
                   if def_map
                   else numpy.zeros((0, 3), dtype='int64'))
        return def_array, def_mask, def_map

    def sentinel_definition(self):
        """An empty definition.

        If you ever need a definition that is syntactically correct but
        doesn't mean a thing, call me.

        """
        return [self._vocab_def.bod, self._vocab_def.eod]
