import sys
import pickle
import gzip
import os
import argparse
import numpy as np


def read_embedding_file(embedding_file):
    """ read embedding file into a dictionary
    deals with GloVe or w2v format as well as pickled dict with single entries
    """

    embeddings = {}
    # pickled dict
    if embedding_file.endswith('.pkl'):
        embeddings = np.load(embedding_file)
        first_array = embeddings.values()[0]
        if type(first_array):# == list and len(first_array) > 1:
            embeddings = {k: v[0] for k, v in embeddings.iteritems()}
        return embeddings 

    # glove or w2v non compressed format
    with open(embedding_file, 'r') as f:
        line = f.readline()
        if len(line.split()) > 2:
            w, v = line.split(' ', 1)
            v = np.fromstring(v, sep=' ')
            embeddings[w] = v
            
        for line in f:
            try:
                w, v = line.split(' ', 1)
            except:
                print w, v
            v = np.fromstring(v, sep=' ')
            embeddings[w] = v
    return embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge 2 embedding files')
    parser.add_argument('embedding_1', type=str,
                        help='First embedding file with missing entries')
    parser.add_argument('embedding_2', type=str,
                        help='Second embedding file')
    parser.add_argument('--restricted', action='store_true',
                        help="Only replace values from embedding_2 by values "
                             "from embedding 1")
    parser.add_argument('output', type=str,
                        help='Output embedding file')


    args = parser.parse_args()

    print "read first file {}".format(args.embedding_1)
    embeddings = read_embedding_file(args.embedding_1)
    print "read second file {}".format(args.embedding_2)
    embeddings_complement = read_embedding_file(args.embedding_2)
    print "merging"
    # merge the 2 embeddings with the first having priority
    if not args.restricted:
        embeddings_complement.update(embeddings)
        embeddings = embeddings_complement
    else:
        new_embeddings = {}
        for k, v in embeddings_complement.iteritems():
            e = embeddings.get(k, None)
            if e is not None:
                new_embeddings[k] = e
            else:
                new_embeddings[k] = v
        embeddings = new_embeddings


    with open(args.output, 'w') as f:
        for key, e in embeddings.iteritems():
            line = key + " "
            line += ' '.join([str(i) for i in e])
            f.write(line + '\n')
