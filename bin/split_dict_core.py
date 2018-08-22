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
    parser = argparse.ArgumentParser("Split a json dictionary in train.json, "
                                     "valid.json, test.json")
    parser.add_argument("--seed", type=int, help="Seed", default=1)

    parser.add_argument("input_dict", help="Dictionary (json)")
    parser.add_argument("output_dir", help="Directory for output dictionaries")
    parser.add_argument("valid_percent", type=float,
                        help="percentage of valid samples")


    args = parser.parse_args()

    print "percent_valid", args.valid_percent
    in_dict = json.load(open(args.input_dict, "r"))

    random.seed(args.seed)
   
    def deduplicate(list_of_lists):
        new_list = []
        for l in list_of_lists:
            if l not in new_list:
                new_list.append(l)
        return new_list

    # 2 similar definitions will have to be in the same split

    keys_to_defs = {k: [" ".join(def_) for def_ in defs] for k, defs in in_dict.iteritems()}
    # compute flattened list of defs
    all_defs = [def_ for defs in keys_to_defs.values() for def_ in defs]
    #all_defs = deduplicate(all_defs[:10000])
    random.shuffle(all_defs)
    all_defs = set(all_defs)
    print "done"
    #random.shuffle(all_defs)
    #all_defs = set(all_defs)
    # deduplicate
    N = len(all_defs)
    print "len defs:", N
    # create a map def -> [w1, w2, ...]
    defs_to_keys = defaultdict(list)
    for w, defs in keys_to_defs.iteritems():
        for def_ in defs:
            defs_to_keys[def_].append(w)
    batches = []

    group_sizes = []
    while len(all_defs) > 0:
        # create new buffers
        new_defs, new_terms = set(), set()
        # init stack of definitions
        stack_defs = set([all_defs.pop()])
        while len(stack_defs) > 0:
            #print "innter loop", len(stack_defs), len(new_terms)
            def_ = stack_defs.pop()
            new_defs.add(def_)
            for k in defs_to_keys[def_]:
                if k in new_terms:
                    continue
                new_terms.add(k)
                defs = keys_to_defs[k]
                for definition in defs:
                    if definition not in stack_defs and definition not in new_defs:
                        stack_defs.add(definition)
        #print "new terms:", new_terms
        #print "new defs:", new_defs
        group_sizes.append((len(new_defs), len(new_terms)))
        for d in new_defs:
            try:
                all_defs.remove(d)
            except:
                pass
        batches.append((new_defs, new_terms, len(new_defs), len(new_terms)))

    dtype = [('defs', list), ('terms', list), ('n_defs', int), ('n_terms',int)]
    batches = np.asarray(batches, dtype=dtype)
    batches = np.sort(batches, order=('n_defs', 'n_terms'))[::-1]
    print "len batches", len(batches)
    terms_train = batches[0][1]
    N_train = len(terms_train)
    batches = batches[1:]

    print "N train", N_train
    N_valid = int((args.valid_percent/100.)*(N - N_train))
    N_test = N - (N_valid + N_train)
    print "rough max numbers:", N_train, N_valid, N_test
    print "total defs left to rpocess", sum([n for _, _, n, _ in batches])
    # compute maximum number of defs in train set and valid
    terms_valid, terms_test = [], []
    to_fill = [(terms_valid, N_valid),
               (terms_test, N_test)]

    i = 0
    counts_defs = [0,0]
    for _, batch, n_defs, _ in batches:
        #print len(all_defs), i, len(to_fill)
        # pick list to fill cyclically
        if len(counts_defs) == 0:
            print "{},{}".format(batch[1],batch[2]),
            continue

        i_split = i%len(to_fill)
        defined_terms, max_N = to_fill[i_split]
        defined_terms.extend(batch)

        counts_defs[i_split] += n_defs
        if max_N <= counts_defs[i_split]:
            print "Done filling split #{}".format(i_split)
            print "contains {} defs".format(counts_defs[i_split])
            del to_fill[i_split]
            del counts_defs[i_split]
            if len(counts_defs) == 0:
                print "Premature end at batch #{}/{}".format(i,len(batches))
                print sum([b[1] for b in batches[i:]])
        i += 1


    terms_splits = [terms_train, terms_valid, terms_test]
    n_defs_in_groups = np.asarray([n_defs for n_defs, _ in group_sizes])
    n_terms_in_groups =  [n_terms for _, n_terms in group_sizes]
    sorted_idx = np.argsort(n_defs_in_groups)[::-1]
    #print n_defs_in_groups
    #plt.hist(n_defs_in_groups[sorted_idx][1:], bins=100)
    #plt.show()
    print "end of algorithm: counts of definitions:", [len(t) for t in terms_splits]
        
    print "stats:"
    print "# (train, valid, test) : {}, {}, {}".format(
        len(terms_splits[0]), len(terms_splits[1]), len(terms_splits[2])
    )
    
    fnames = ["train", "valid", "test"]
    fnames = [os.path.join(args.output_dir, fn+".json") for fn in fnames]
    # check that none of the files already exist, don't want to mistakenly
    # erase anything
    for fname in fnames:
        assert(not os.path.isfile(fname))

    for terms, fname in zip(terms_splits, fnames):
        with open(fname, 'w') as dst:
            dict_ = {t: in_dict[t] for t in terms}
            json.dump(dict_, dst, indent=2)

if __name__ == "__main__":
    main()
