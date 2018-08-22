#!/usr/bin/env python

from __future__ import division

import math
import argparse
import logging
import json
from nltk.corpus import stopwords, wordnet
from dictlearn.vocab import Vocabulary
from collections import defaultdict

def main():
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Transform a dictionary")
    parser.add_argument("--remove_words_from", 
                        help="remove words from an external list (\\n separated)")

    parser.add_argument("--remove_stop_words", action='store_true',
                        help="remove stopwords from the dict")
    parser.add_argument("--remove_mwe", action='store_true',
                        help="remove multiword expressions from the dict")
    parser.add_argument("--retrieve_original_case", action='store_true',
                        help="retrieve original case (when entries are all "
                             "lowercased")
    parser.add_argument("input_dict", help="Dictionary")
    parser.add_argument("output_dict", help="Output dictionary")
    args = parser.parse_args()

    dict_ = json.load(open(args.input_dict, "r"))
    
    # assert we are doing one of the optional preprocessing at least
    assert(args.remove_stop_words or args.retrieve_original_case
           or args.remove_mwe or args.remove_words_from)

    if args.remove_words_from:
        blacklist = set()
        with open(args.remove_words_from, 'r') as f:
            for l in f:
                blacklist.add(l.strip())
                
        tmp_dict = {v: k for (v,k) in dict_.iteritems() if v not in blacklist}
        print "deleted {} words from the blacklist".format(len(dict_) - len(tmp_dict))
        dict_ = tmp_dict

    if args.remove_stop_words:
        stop_words = stopwords.words('english')
        tmp_dict = {v: k for (v,k) in dict_.iteritems() if v not in stop_words}
        print "deleted {} stop words".format(len(dict_) - len(tmp_dict))
        dict_ = tmp_dict

    if args.remove_mwe:
        tmp_dict = {v: k for (v,k) in dict_.iteritems() if '_' not in v}
        print "deleted {} multi word expressions".format(len(dict_) - len(tmp_dict))
        dict_ = tmp_dict

    if args.retrieve_original_case:
        tmp_dict = defaultdict(list)
        for w, defs in dict_.iteritems():
            synsets = wordnet.synsets(w)
            assert(len(defs) == len(synsets))
            for synset, def_ in zip(synsets, defs):
                cased_lemma = [c for c in synset.lemma_names() if c.lower() == w]
                if len(cased_lemma) == 0:
                    cased_lemma = [w]
                # There are a couple of cases like alphabet letters, sun, moon,
                # earth, doomsday,... which can be written both with and
                # without caps
                if len(cased_lemma) > 1:
                    print cased_lemma, w
                for c in cased_lemma:
                    tmp_dict[c].append(def_)
        dict_ = tmp_dict

    with open(args.output_dict, 'w') as dst:
        json.dump(dict_, dst, indent=2)

if __name__ == "__main__":
    main()



