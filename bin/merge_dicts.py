#!/usr/bin/env python

from __future__ import division

import math
import argparse
import logging
import json
from collections import defaultdict

def main():
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Merge dictionaries")
    parser.add_argument("input_dicts", help="Dictionary files separated by a comma")
    parser.add_argument("output_dict", help="Output dictionary")
    args = parser.parse_args()

    dictionaries_fn = args.input_dicts.split(',')
    assert(len(dictionaries_fn) > 1)
    dict_ = {}
    for fn_ in dictionaries_fn:
        dict_.update(json.load(open(fn_, "r")))

    with open(args.output_dict, 'w') as dst:
        json.dump(dict_, dst, indent=2)

if __name__ == "__main__":
    main()



