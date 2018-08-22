import sys
import pickle
import gzip
import os
import argparse
import numpy as np
from dictlearn.vocab import Vocabulary

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
    parser = argparse.ArgumentParser(description='Write the list of words in embeddings but not in dict vocabulary')
    parser.add_argument('embeddings', type=str)
    parser.add_argument('vocabulary', type=str)
    parser.add_argument('vocabulary_counts', type=str)
    parser.add_argument('absent_words', type=str)

    args = parser.parse_args()

    print "read first file {}".format(args.embeddings)
    embeddings = read_embedding_file(args.embeddings)
    print "read vocabulary file {}".format(args.vocabulary)
    vocabulary = Vocabulary(args.vocabulary)
    print "read vocabulary for counts estimation file {}".format(args.vocabulary_counts)
    vocabulary_counts = Vocabulary(args.vocabulary_counts)

    vocabulary = set(vocabulary.words) # faster lookup

    absent_in_vocab = set([w for w in embeddings.keys() if w not in vocabulary])
    print("Number of absent words in vocab", len(absent_in_vocab))
    absent_in_vocab = sorted(list(absent_in_vocab), key=lambda w: vocabulary_counts.word_freq(w), reverse=True)

    with open(args.absent_words, 'w') as f:
        for w in absent_in_vocab:
            f.write(w.encode('utf8') + '\n')
