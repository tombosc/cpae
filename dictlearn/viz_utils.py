"""
Utilities used for visualisation and analysis of the results
"""

from scipy.stats import spearmanr, mannwhitneyu
from random import shuffle
import numpy as np
from scipy.stats import rankdata
from web.evaluate import cosine_similarity
import matplotlib.pyplot as plt
import json

def load_dict(filename):                   
    return json.load(open(filename, "r"))

def partial_correlation(x, y, z):
    """
    return partial correlation coefficient between x and y, controlling for z
    """
    rho_xy = spearmanr(x, y).correlation
    rho_xz = spearmanr(x, z).correlation
    rho_zy = spearmanr(z, y).correlation

    rho_xy_z = ((rho_xy - rho_xz * rho_zy) /
                ((np.sqrt(1-(rho_xz)**2))*(np.sqrt(1-(rho_zy)**2))))
    return rho_xy_z

def print_coverage(data, emb):
    """
    print coverage of embeddings emb on the dataset data
    """
    print "Coverage of the embedding on the dataset:"
    not_found = []
    n_total = 0
    pairs = 0
    for x1, x2 in data.X:
        x1_not_found, x2_not_found = False, False
        if emb.get(x1) is None:
            not_found.append(x1)
            x1_not_found = True
        if emb.get(x2) is None:
            not_found.append(x2)
            x2_not_found = True
        if x1_not_found and x2_not_found:
            pairs += 1
        n_total+=2
    print "Total and found words: {}, {}".format(
        n_total, n_total-len(not_found)
    )
    print "# of pairs where the 2 words are missing vs total # of pairs:",
    print pairs, ", ", n_total/2
    shuffle(not_found)
    print "Not found (excerpt):", not_found[:20]

def cosine_sim(emb, w1, w2, default):
    """
    wraps WEB cosine similarity to be compatible with poly embeddings
    """
    try:
        t1 = emb.get_multi(w1, default)
        t2 = emb.get_multi(w2, default)
        model = 'AvgSim'
    except:
        t1 = emb.get(w1, default)
        t2 = emb.get(w2, default)
        t1 = t1.reshape((1,-1))
        t2 = t2.reshape((1,-1))
        model = None
    return cosine_similarity(t1, t2, model)

def compute_ranks(data, emb, default=None):
    """
    returns pairs of ranks (predictions, ground_truth)
    
    default: if None, takes the mean of all vectors
             else, string indicating the word in the vocab that's the UNK token
    """
    #subset = np.random.randint(low=0, high=len(data.X), size=n_samples)    
    scores = []
    if not default:
        default_vec = np.mean(emb.vectors, axis=0).reshape((1,-1))
    else:
        default_vec = emb.get(default)

    for w1, w2 in data.X:
        scores.append(cosine_sim(emb, w1, w2, default_vec))

    ranks_predicted = rankdata(scores)
    ranks_truth = rankdata(data.y)
    return ranks_predicted, ranks_truth

def plot_hist_diff_ranks(diff_ranks):
    """ return a figure with histogram of rank differences """
    fig = plt.figure()
    plt.hist(diff_ranks, bins=10)
    plt.title("histogram of rank differences")
    return fig
   
def str_spearman(data1, data2):
    assert(len(data1) == len(data2))
    s = "# datapoints {} ; ".format(len(data1))
    rho, p = spearmanr(data1, data2)
    s += "\rho = {:0.3f} ; p = {:0.2E}".format(rho, p)
    return s

def str_mann_whitney(data1, data2):
    s = "# datapoints {}, {} ; ".format(len(data1), len(data2))
    U, p = mannwhitneyu(data1, data2, alternative='two-sided')
    s += "U = {:0.2f} ; p = {:0.2E}".format(U, p)
    return s


def print_error_vs_num_defs(data, dict_, diff_ranks):
    abs_diff_ranks = np.abs(diff_ranks)

    print "Spearman correlations between errors and number of definitions:"
    get_n_defs = lambda word: len(dict_[word]) if word in dict_ else 0
    n_defs = [get_n_defs(w1) + get_n_defs(w2) for w1, w2 in data.X]
    n_defs_1 = [get_n_defs(w1) for w1, _ in data.X]
    n_defs_2 = [get_n_defs(w2) for _, w2 in data.X]
    #min_n_defs = [min(n1, n2) for n1,n2 in zip(n_defs_1, n_defs_2)]
    #max_n_defs = [max(n1, n2) for n1,n2 in zip(n_defs_1, n_defs_2)]
    abs_diff_defs = [np.abs(n1-n2) for n1,n2 in zip(n_defs_1, n_defs_2)]


    for var, name_var in [(n_defs, 'number of defs'),
                          #(min_n_defs, 'minimum number of defs'),
                          #(max_n_defs, 'max number of defs'),
                          (abs_diff_defs, 'abs diff of number of defs')]:
        print "spearman coefficient between diff_ranks and "+ name_var
        print str_spearman(diff_ranks, var)
        print "spearman coefficient between abs_diff_ranks and "+ name_var
        print str_spearman(abs_diff_ranks, var)

def spearman_train_test(data, dict_test, rank_model):
    in_test = lambda w: w in dict_test
    at_least_one_in_test = np.asarray([int(in_test(w1) or in_test(w2)) for w1, w2 in data.X])
    both_in_test = np.asarray([int(in_test(w1) and in_test(w2)) for w1, w2 in data.X])

    prediction_train = [r for i, r in enumerate(rank_model) if at_least_one_in_test[i] == 0]
    prediction_test = [r for i, r in enumerate(rank_model) if at_least_one_in_test[i] > 0]
    prediction_test_two = [r for i, r in enumerate(rank_model) if both_in_test[i]]

    gt_train = [y for i, y in enumerate(data.y) if at_least_one_in_test[i] == 0]
    gt_test = [y for i, y in enumerate(data.y) if at_least_one_in_test[i] > 0]
    gt_test_two = [y for i, y in enumerate(data.y) if both_in_test[i]]

    print "number of pairs which contains at least a word in the test set:", sum(at_least_one_in_test)
    print "number of pairs which contains both words in the test set:", sum(both_in_test)
    print "global spearman coeff:", str_spearman(rank_model, data.y)
    print "spearman in train:",
    print str_spearman(prediction_train, gt_train)
    print "spearman in test:",
    print str_spearman(prediction_test, gt_test)
    print "spearman both in test:",
    print str_spearman(prediction_test_two, gt_test_two)



def print_plot_error_vs_in_test(data, dict_, dict_test, diff_ranks, name, ax=None):
    abs_diff_ranks = np.abs(diff_ranks)

    print "Correlations between errors and words being defined in test set"
    in_test = lambda w: w in dict_test
    #not_in_train = lambda w: int(w not in dict_) 
    at_least_one_in_test = np.asarray([int(in_test(w1) or in_test(w2)) for w1, w2 in data.X])
    both_in_test = np.asarray([int(in_test(w1) and in_test(w2)) for w1, w2 in data.X])

    diff_in_train = [diff for i, diff in enumerate(diff_ranks) if at_least_one_in_test[i] == 0]
    diff_in_test = [diff for i, diff in enumerate(diff_ranks) if at_least_one_in_test[i] > 0]

    abs_diff_in_train = [diff for i, diff in enumerate(abs_diff_ranks) if at_least_one_in_test[i] == 0]
    abs_diff_in_test = [diff for i, diff in enumerate(abs_diff_ranks) if at_least_one_in_test[i] > 0]

    print "At least one in test:"
    print "Mann-Whitney U test:", str_mann_whitney(diff_in_train, diff_in_test)
    print "Mann-Whitney U test:", str_mann_whitney(abs_diff_in_train, abs_diff_in_test)

    N_train = len(diff_in_train)
    N_test = len(diff_in_test)
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    data_boxplot=[abs_diff_in_train, abs_diff_in_test]
    labels_boxplot=[r'$|\delta$|(train)' + "\n" + r'$n=' + str(N_train) + r'$',
                    r'$|\delta$|(test)' + "\n" + r'$n=' + str(N_test) + r'$']
    ax.set_title(name)
    ax.boxplot(data_boxplot, labels=labels_boxplot)
    ax.legend()
    #return fig

def print_plot_error_vs_in_frequency(data, vocab_corpus, diff_ranks, name, ax=None):
    abs_diff_ranks = np.abs(diff_ranks)
    print "Correlations between errors and frequency of words"

    def average_count_def(w):
        counts = []
        if not w in dict_:
            return 0
        defs_count = []
        for def_ in dict_[w]:
            defs_count.append(np.mean([vocab_corpus.word_freq(i) for i in def_]))
        #print defs_count, dict_[w]
        return np.max(defs_count)

    mean_counts_all_defs = np.mean([average_count_def(w) for w in dict_.keys()])

    print average_count_def('state')
    print average_count_def('yellow')
    print average_count_def('veer')
    print average_count_def('uvea')

    average_counts = [np.mean([average_count_def(w1), average_count_def(w2)]) - mean_counts_all_defs for w1, w2 in data.X]

    print "spearman diffranks, avg counts:", spearmanr(diff_ranks, average_counts)
    print "spearman abs(diffranks), avg counts:", spearmanr(abs_diff_ranks, average_counts)

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.scatter(abs_diff_ranks, average_counts, color="blue", label="average_counts", alpha=0.4)
    #plt.scatter(diff_ranks, n_defs_1, color="green", label="n_defs_1", alpha=0.2)
    #plt.scatter(diff_ranks, n_defs_2, color="red", label="n_defs_2", alpha=0.2)
    ax.legend()
    ax.set_title(name)

def print_plot_error_vs_frequency(data, vocab_corpus, diff_ranks, name, ax=None, abs_=True):
    abs_diff_ranks = np.abs(diff_ranks)

    print "Correlations between errors and frequency of words"
    count = lambda w: vocab_corpus.word_freq(w)
    counts_words = [(count(w1), count(w2)) for w1, w2 in data.X]
    n_unks = 0
    for counts in enumerate(counts_words):
        n_unks += int(counts[0] == 0 or counts[1] == 0)
    print "# pairs which contains at least 1 unk: {}/{}".format(n_unks, len(data.X))

    # (word_freq actually returns a count, not a frequency)
    # UNKs are fine: vocab.word_freq returns 0 if not found.
    mean_counts = [np.mean(np.log(count(w1) + 1) + np.log(count(w2) + 1)) for w1, w2 in data.X]

    if abs_:
        diff = abs_diff_ranks
        print "spearman abs(diffranks), mean counts:", str_spearman(abs_diff_ranks, mean_counts)
    else:
        diff = diff_ranks
        print "spearman diffranks, mean counts:", str_spearman(diff_ranks, mean_counts)

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
   
    #ax.xaxis.set_label_position('top') 
    ax.scatter(mean_counts, diff, color="blue", label="sum counts", s=10, alpha=0.4)
    ax.set_title(name)

def print_error_vs_len_defs(data, dict_, diff_ranks):       
    abs_diff_ranks = np.abs(diff_ranks)

    print "Correlations between errors and length of definitions"

    def mean_len_defs(pairs):
        mean_len = []
        for w1, w2 in pairs:
            if w1 in dict_:
                v1 = np.mean([len(def_) for def_ in dict_[w1]])
            else:
                v1 = 0
            if w2 in dict_:
                v2 = np.mean([len(def_) for def_ in dict_[w2]])
            else:
                v2 = 0
            mean_len.append(np.mean([v1, v2]))
        return mean_len
        
    mean_len = mean_len_defs(data.X)

    print "spearman diffranks, mean length of def:", str_spearman(diff_ranks, mean_len)
    print "spearman abs(diffranks), mean length of def:", str_spearman(abs_diff_ranks, mean_len)

def print_error_vs_in_vocabulary_defs(data, vocab_defs, diff_ranks):
    abs_diff_ranks = np.abs(diff_ranks)

    print "Correlations between errors and presence of words in the vocabulary of definitions"
    is_in_vocab = lambda w: vocab_defs.word_to_id(w) != vocab_defs.unk
    both_in_vocab = np.asarray([int(is_in_vocab(w1) and is_in_vocab(w2)) for w1, w2 in data.X])
    at_least_one_in_vocab = np.asarray([int(is_in_vocab(w1) or is_in_vocab(w2)) for w1, w2 in data.X])

    print "spearman diffranks, both defs are in vocab:", str_spearman(diff_ranks, both_in_vocab)
    print "spearman abs(diffranks), both defs are in vocab:", str_spearman(abs_diff_ranks, both_in_vocab)
    print "spearman diffranks, at least one def is in vocab:", str_spearman(diff_ranks, at_least_one_in_vocab)
    print "spearman abs(diffranks), at least one def is in vocab:", str_spearman(abs_diff_ranks, at_least_one_in_vocab)


def print_error_vs_avg_count_def(data, dict_, vocab_defs, diff_ranks, name):
    abs_diff_ranks = np.abs(diff_ranks)
    print "Correlations between errors and geometric average of counts in definitions (avg over words, sentences, words in the pair)"

    def average_frequency(w):
        if not w in dict_:
            return 0
        m = []
        for def_ in dict_[w]:
            m.append(np.mean([-np.log(vocab_defs.word_freq(i) + 1) for i in def_]))
        return np.mean(m)

    avg_freqs = [np.mean([average_frequency(w1), average_frequency(w2)]) for w1, w2 in data.X]
    print "spearman diffranks, avg counts:", str_spearman(diff_ranks, avg_freqs)
    print "spearman abs(diffranks), avg counts:", str_spearman(abs_diff_ranks, avg_freqs)
