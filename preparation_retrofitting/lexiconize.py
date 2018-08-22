#!/usr/bin/env python
""" Create a lexicon for retrofitting out of a dictionary (json dictionary).
"""

import h5py
import argparse
import logging
from six import text_type
import json
import collections
from nltk.corpus import stopwords


def main():
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Builds a lexicon")
    parser.add_argument("dictionary", help="Input dictionary")
    parser.add_argument("lexicon", help="Output lexicon")
    args = parser.parse_args()

    lexicon_lines = []
    logging.info("Will build the lexicon from definitions in the dictionary")
    dict_ = json.load(open(args.dictionary, "r"))

    valid_word = lambda w: not (w in stopwords.words('english')
                            or '_' in w
                            or w in ['(',')',';', '\'s', 'e.g.', ',', '\'', '`'])

    for word, list_defs in dict_.iteritems():
        s = word
        if not valid_word(word):
            continue
        empty_def = True
        for def_ in list_defs:
            for def_word in def_:
                if not valid_word(def_word):
                    continue
                empty_def = False
                s += ' ' + def_word
        if empty_def:
            continue
        s += '\n'
        lexicon_lines.append(s)

    with open(args.lexicon, "w") as f:
        for l in lexicon_lines:
            f.write(l.encode('utf8'))

if __name__ == "__main__":
    main()

