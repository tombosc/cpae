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
    parser = argparse.ArgumentParser(description='Transform pkl to txt')
    parser.add_argument('embedding_in', type=str)
    parser.add_argument('embedding_out', type=str)

    args = parser.parse_args()

    print "read first file {}".format(args.embedding_in)
    embeddings = read_embedding_file(args.embedding_in)

    with open(args.embedding_out, 'w') as f:
        for key, e in embeddings.iteritems():
            line = key + " "
            line += ' '.join([str(i) for i in e])
            f.write(line.encode('utf8') + '\n')
