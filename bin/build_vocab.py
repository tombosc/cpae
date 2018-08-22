#!/usr/bin/env python
"""
Builds vocabulary from a dictionary

Call as:
python bin/build_vocab.py $DATA_DIR/snli/dict_all_3_05_lowercase_lemma.json
    $DATA_DIR/snli/dict_all_3_05_lowercase_lemma_vocab.txt
"""

import h5py
import argparse
import logging
from six import text_type
import json
import collections

from dictlearn.vocab import Vocabulary


def main():
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Builds a vocabulary")
    parser.add_argument("--top-k", type=int, help="Top most frequent words to leave")
    parser.add_argument("--keys-only", action='store_true', help="Build vocab of all keys")
    parser.add_argument("--with-keys", action='store_true', 
                        help="Count keys and words in definitions")
    parser.add_argument("dictionary", help="Input dictionary")
    parser.add_argument("vocabulary", help="Output vocabulary")
    args = parser.parse_args()

    text = []
    if args.dictionary.endswith('.json'):
        text = collections.defaultdict(int)
    for f_name in args.dictionary.split(","):
        logging.info("Processing " + f_name)
        assert(f_name.endswith('.json'))
        logging.info("Will build the vocabulary from definitions in a dictionary")
        dict_ = json.load(open(f_name, "r"))
        for word, list_defs in dict_.items():
            if args.keys_only or args.with_keys:
                text[word] += 1
            if not args.keys_only:
                for def_ in list_defs:
                    for def_word in def_:
                        text[def_word] += 1

        logging.info("{} words".format(len(text)))

    vocab = Vocabulary.build(text, args.top_k)
    vocab.save(args.vocabulary)

if __name__ == "__main__":
    main()

