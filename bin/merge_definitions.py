#!/usr/bin/env python

from __future__ import division
import os.path
import argparse
import random
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser("Merge all the definitions")
    parser.add_argument("--seed", type=int, help="Seed", default=1)
    parser.add_argument("--separator", help="Separator token", default="<sep>", type=str)
    parser.add_argument("input_dict", help="Input dictionary (json)")
    parser.add_argument("output_dict", help="Output dictionary")

    args = parser.parse_args()

    in_dict = json.load(open(args.input_dict, "r"))
    out_dict = {}

    print args.separator.join(["a", "b"])

    random.seed(args.seed)

    for k, v in in_dict.iteritems():
        random.shuffle(v)
        # in regular dictionaries, first definition matches the most frequent
        # sense. But in WN, senses are split according to POS
        # we want diversity in POS
        defs = [def_ + [args.separator] for def_ in v]
        concatenated_defs = [w for def_ in defs for w in def_]
        concatenated_defs = concatenated_defs[:-1] # remove last separator
        out_dict[k] = [concatenated_defs]

    with open(args.output_dict, 'w') as dst:
        json.dump(out_dict, dst, indent=2)

if __name__ == "__main__":
    main()
