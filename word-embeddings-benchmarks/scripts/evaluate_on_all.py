#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 This script calculates embedding results against all available fast running
 benchmarks in the repository and saves results as single row csv table.

 Usage: ./evaluate_on_all -f <path to file> -o <path to output file>

 NOTE:
 * script doesn't evaluate on WordRep (nor its subset) as it is non standard
 for now and long running (unless some nearest neighbor approximation is used).

 * script is using CosAdd for calculating analogy answer.

 * script is not reporting results per category (for instance semantic/syntactic) in analogy benchmarks.
 It is easy to change it by passing category parameter to evaluate_analogy function (see help).
"""
from optparse import OptionParser
import logging
import os
from web.embeddings import fetch_GloVe, load_embedding
from web.datasets.utils import _get_dataset_dir

from web.evaluate import evaluate_on_all, evaluate_on_all_multi


# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="Path to the file with embedding. If relative will load from data directory.",
                  default=None)

parser.add_option("-p", "--format", dest="format",
                  help="Format of the embedding, possible values are: word2vec, word2vec_bin, dict and glove.",
                  default=None)

parser.add_option("-o", "--output", dest="output",
                  help="Path where to save results.",
                  default=None)

parser.add_option("-c", "--clean_words", dest="clean_words",
                  help="Clean_words argument passed to load_embedding function. If set to True will remove"
                       "most of the non-alphanumeric characters, which should speed up evaluation.",
                  default=False)

parser.add_option("-m", "--multi", dest="multi_prototype", action='store_true',
                  help="Evaluate as multi-prototype embeddings")

parser.add_option("-l", "--lowercase", dest="lowercase", action='store_true',
                  help="Lowercase")

parser.add_option("-n", "--normalize", dest="normalize", action='store_true',
                  help="Normalize")


parser.add_option("-s", "--model", dest="model", default='MaxSim',
                  help="Similarity evaluation model: either MaxSim or AvgSim")

parser.add_option("--lower-or-lemma", dest="lower_or_lemma", action='store_true',
                  help="Lowercase and then lemmatize to try to find embeddings of OOV words")

parser.add_option("--only-sim-rel", dest="only_sim_rel", action='store_true',
                  help="Only use similarity and relatedness benchmarks")


if __name__ == "__main__":
    (options, args) = parser.parse_args()

    # Load embeddings
    fname = options.filename
    if not fname:
        w = fetch_GloVe(corpus="wiki-6B", dim=300)
    else:
        if not os.path.isabs(fname):
            fname = os.path.join(_get_dataset_dir(), fname)

        format = options.format

        if not format:
            _, ext = os.path.splitext(fname)
            if ext == ".bin":
                format = "word2vec_bin"
            elif ext == ".txt":
                format = "word2vec"
            elif ext == ".pkl":
                format = "dict"

        assert format in ['word2vec_bin', 'word2vec', 'glove', 'bin', 'dict', 'dict_poly'], "Unrecognized format"

        load_kwargs = {}
        if format == "glove":
            vocab_size = sum(1 for line in open(fname))
            dim = len(next(open(fname)).split()) - 1
            load_kwargs={'dim': dim, 'vocab_size': vocab_size}

        w = load_embedding(fname, format=format, normalize=options.normalize, 
                           clean_words=options.clean_words,
                           lower=options.lowercase,
                           lowercase_if_OOV=options.lower_or_lemma,
                           lemmatize_if_OOV=options.lower_or_lemma,
                           load_kwargs=load_kwargs)

    out_fname = options.output if options.output else "results.csv"

    if options.multi_prototype:
        results = evaluate_on_all_multi(w, options.model)
    else:
        results = evaluate_on_all(w, only_sim_rel=options.only_sim_rel)

    logger.info("Saving results...")
    print(results)
    results.to_csv(out_fname)
