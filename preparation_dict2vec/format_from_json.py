#!/usr/bin/env python

from __future__ import division

import math
import argparse
import logging
import json
from nltk.corpus import stopwords, wordnet
from collections import defaultdict

def main():
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Transform dictionary format from json"
                                     "to dict2vec. Remove stopwords.")

    parser.add_argument("input_dict", help="Dictionary file in json format")
    parser.add_argument("dict_name", help="Dictionary name")
    parser.add_argument("output_dict", help="Output dict in dict2vec format")
    args = parser.parse_args()

    dict_ = json.load(open(args.input_dict, "r"))

    with open(args.output_dict, 'w') as f:
        for w, defs in dict_.iteritems():
            line = args.dict_name + " " + w + " " # name of the dictionary
            # first remove stopwords
            for def_ in defs:
                def_ = [w for w in def_ if w not in stopwords.words('english')]
                line += ' '.join(def_) + '. '
            f.write(line.encode('utf8') + '\n')


if __name__ == "__main__":
    main()
