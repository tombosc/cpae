#!/usr/bin/env python

import os.path
import argparse
import random
import json
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser("Extract defs from a json dictionary "
                                     "into a text file")
    parser.add_argument("input_dict", help="Dictionary (json)")
    parser.add_argument("output_text", help="Output (txt)")
    parser.add_argument("--deduplicate", action='store_true')
    parser.add_argument("--include-key", action='store_true')

    args = parser.parse_args()

    in_dict = json.load(open(args.input_dict, "r"))
   
    def deduplicate(list_of_lists):
        new_list = []
        for l in list_of_lists:
            if l not in new_list:
                new_list.append(l)
        return new_list

    if args.include_key:
        all_defs = [[k] + def_ for k, defs in in_dict.iteritems() for def_ in defs]
    else:
        all_defs = [def_ for defs in in_dict.values() for def_ in defs]
    if args.deduplicate:
        all_defs = deduplicate(all_defs)

    with open(args.output_text, 'w') as f:
        for d in all_defs:
            f.write(' '.join([e.encode('utf-8') for e in d]) + '\n')

if __name__ == "__main__":
    main()
