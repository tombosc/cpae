#!/usr/bin/env python

import argparse
import logging
import numpy as np

from os import path
from dictlearn.vocab import Vocabulary

def main():
    logging.basicConfig(
        level='INFO',
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("Converts GLOVE embeddings to a numpy array")
    parser.add_argument("txt", help="GLOVE data in txt format")
    parser.add_argument("npy", help="Destination for npy format")
    parser.add_argument("vocab_out", help="output vocabulary")
    parser.add_argument("--vocab", default="", help="Performs subsetting based on passed vocab")

    # OOV handling
    parser.add_argument("--try-lowercase", action="store_true", help="Try lowercase")

    args = parser.parse_args()

    if args.vocab == "":
        raise NotImplementedError("Not implemented")
        embeddings = []
        dim = None
        with open(args.txt) as src:
            for i, line in enumerate(src):
                tokens = line.strip().split()
                features = map(float, tokens[1:])
                dim = len(features)
                embeddings.append(features)
                if i and i % 100000 == 0:
                    print i
        embeddings = [[0.] * dim] * len(Vocabulary.SPECIAL_TOKEN_MAP) + embeddings
        np.save(args.npy, embeddings)
    else:
        vocab = Vocabulary(args.vocab)
            
        print('Computing GloVe')

        # Loading
        embeddings_index = {}
        f = open(args.txt)

        print('Reading GloVe file')
        for line in f:
            values = line.split(' ')
            word = values[0]
            dim = len(values[1:])
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

        f.close()

        # Embedding matrix: larger than necessary
        f_out = open(args.vocab_out, 'w')
        n_specials = len(Vocabulary.SPECIAL_TOKEN_MAP.values())

        embedding_matrix = np.zeros((vocab.size() + n_specials, dim))
        for special_token in Vocabulary.SPECIAL_TOKEN_MAP.values(): 
            line = '<' + special_token + '>' + " 0\n"
            f_out.write(line.encode('utf-8'))

        i = n_specials
        #i = 0
        for word, count in zip(vocab.words, vocab.frequencies):
            embedding_vector = embeddings_index.get(word)
            if args.try_lowercase and not isinstance(embedding_vector, np.ndarray):
                embedding_vector = embeddings_index.get(word.lower())
            in_glove = embedding_vector is not None
            last_comp = None
            if in_glove:
                last_comp = embedding_vector[-1]
            #print "i: {}, word {}, count {}, in_glove {}, last {}".format(i, word, count, in_glove, last_comp)
            if in_glove:
                try:
                    embedding_matrix[i] = embedding_vector
                except:
                    print "error idx", i
                # else, null vector 
                #print "writing:", line, i
                line = word + " " + str(count) + "\n"
                f_out.write(line.encode('utf-8'))
                i += 1
            if i and i%10000==0:
                print "i:", i
        f_out.close()
        np.save(args.npy, embedding_matrix[:i])

if __name__ == "__main__":
    main()
