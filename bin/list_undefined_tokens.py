#!/usr/bin/env python
"""
List all undefined tokens that appear in definition but are not defined.
"""

import argparse
import logging
import json
import collections

from dictlearn.vocab import Vocabulary


def main():
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("List undefined tokens")
    parser.add_argument("vocabulary", help="Input vocabulary")
    parser.add_argument("dictionary", help="Input dictionary")
    args = parser.parse_args()

    undefined_tokens_and_freqs = []
    vocab = Vocabulary(args.vocabulary)
    with open(args.dictionary) as f:
        dictionary = json.load(f)
    for w, c in zip(vocab.words, vocab.frequencies):
        if w not in dictionary.keys():
            undefined_tokens_and_freqs.append((w, c))

    undefined_tokens_and_freqs = sorted(undefined_tokens_and_freqs, key=lambda x: x[1], reverse=True)
    for w, c in undefined_tokens_and_freqs:
        print(w)

if __name__ == "__main__":
    main()

