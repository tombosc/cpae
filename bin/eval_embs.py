import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import numpy as np
import argparse
import os
import functools
import fuel

from web.datasets.analogy import fetch_google_analogy
from web.datasets.similarity import (fetch_MEN, fetch_SimVerb3500,
    fetch_SimLex999, fetch_RW, fetch_SCWS, fetch_MTurk, fetch_WS353)
from web.embeddings import load_embedding
from web.evaluate import evaluate_similarity

from dictlearn.vocab import Vocabulary
from dictlearn.viz_utils import (partial_correlation, print_coverage, compute_ranks,
                                 plot_hist_diff_ranks, print_error_vs_num_defs, 
                                 print_plot_error_vs_in_test, print_error_vs_in_vocabulary_defs,
                                 print_error_vs_len_defs, print_plot_error_vs_frequency,
                                 print_error_vs_avg_count_def, load_dict, spearman_train_test)

if __name__ == "__main__":

    plt.style.use('ggplot')
    np.random.seed(0)

    # Configure logging
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

    # SETUP
    datasets = [('SL999', fetch_SimLex999()),
                ('SL333', fetch_SimLex999(which='333')),
                ('SV3500-t', fetch_SimVerb3500(which='test')),
                ('WS353', fetch_WS353()),
                #('WS353R', fetch_WS353(which='relatedness')),
                #('RW', fetch_RW()),
                ('MEN-t', fetch_MEN(which='test')),
                ('SCWS', fetch_SCWS()),
                ('MTurk', fetch_MTurk())]
    vocab_defs_fname = os.path.join(fuel.config.data_path[0], "vocab.txt")
    logging.info("using vocab for definition {}".format(vocab_defs_fname))
    # END SETUP

    parser = argparse.ArgumentParser("Evaluate embeddings")
    parser.add_argument("emb_filename", help="Location of embeddings")
    parser.add_argument("emb_format", help="either 'glove', 'dict' or 'dict_poly'")
    parser.add_argument("root_dicts", 
                         help="dirname that contains all.json and test.json")
    parser.add_argument("--lowercase", action='store_true')
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--vocab_size", default=None, type=int,
                        help="vocab size (GloVe only)")

    args = parser.parse_args()
    model_name = args.emb_filename.split('/')[-2]

    is_dict_embedding = args.emb_format.startswith('dict')        

    if not is_dict_embedding and not args.vocab_size:
        raise ValueError("GloVe embeddings require a vocab size")

    if args.lowercase or args.normalize:
        raise NotImplementedError('Bug')

    kwargs_emb = {}
    if is_dict_embedding: #unavailable for GloVe
        kwargs_emb = {"normalize": args.normalize,
                      "lowercase": args.lowercase}
    else:
        kwargs_emb = {"dim": 300,
                      "vocab_size": args.vocab_size}
    emb = load_embedding(args.emb_filename, format=args.emb_format,
                         load_kwargs=kwargs_emb, lowercase_if_OOV=False, 
                         lemmatize_if_OOV=False, normalize=False)


    model_name = args.emb_filename.split('/')[-2]
    # TODO: need to feed dim and vocab_size? or useless?
  
    vocab_defs, dict_, test_dict = None, None, None
    if is_dict_embedding:
        vocab_defs = Vocabulary(vocab_defs_fname)
        fname_dict = os.path.join(args.root_dicts, "all.json")
        fname_test_dict = os.path.join(args.root_dicts, "test.json")
        dict_ = load_dict(fname_dict)
        test_dict = load_dict(fname_test_dict)

    dirname = os.path.join('results/figures/', model_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    diff_ranks = []
    for name, data in datasets:
        print "dataset:", name
        print_coverage(data, emb)
        print ""
        rank_model, rank_truth = compute_ranks(data, emb)
        diff_ranks.append(rank_model-rank_truth)
        fig = plot_hist_diff_ranks(diff_ranks[-1])
        spearman_train_test(data, test_dict, rank_model)
        fig.savefig(os.path.join(dirname, "diff_ranks_histogram_" + name + ".pdf"))

    def iterate(callback, dict_specific, plot_fname=None, one_figure=False):
        i = 0
        if one_figure:
            n_row = 2
            n_col = len(datasets) / n_row
            fig, axes = plt.subplots(n_row, n_col, figsize=(10,6))

        for diff, data in zip(diff_ranks, datasets):
            name, data = data
            if not dict_specific or args.emb_format.startswith('dict'):
                print "On", name
                kwargs = {'diff_ranks': diff,
                          'data': data,
                          'name': name}

                if one_figure:
                    if n_row == 1:
                        kwargs['ax'] = axes[i]
                    else:
                        kwargs['ax'] = axes[i/n_col, i%n_col]
                    callback(**kwargs)
                else:
                    fig = callback(**kwargs)

                if plot_fname and not one_figure:
                    fig.savefig(os.path.join(dirname, plot_fname + "_" + name + ".pdf"))
                print ""

            i+=1
        if plot_fname and one_figure:
            fig.tight_layout()
            fig.savefig(os.path.join(dirname, plot_fname + "_" + model_name + ".pdf"))
