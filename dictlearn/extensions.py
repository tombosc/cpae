import os
import signal
import time
import subprocess
import atexit
import logging
import cPickle

try:
    import pandas as pd
except ImportError:
    pass
import scipy
import numpy
import numpy as np

import theano
from theano.gof.graph import io_toposort
from theano.scan_module.scan_op import Scan

import ssl
if hasattr(ssl, '_create_unverified_context'):
    print("Disabling SSL verification in HTTP")
    ssl._create_default_https_context = ssl._create_unverified_context

from collections import defaultdict
from six import iteritems

from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import MonitoringExtension
from blocks.serialization import load, load_parameters
from blocks.extensions.saveload import Load, Checkpoint
from blocks.graph import ComputationGraph

from dictlearn.util import get_free_port

logger = logging.getLogger(__name__)

def construct_dict_embedder(theano_fnc, vocab, retrieval):
    """
    Parameters
    ----------
    theano_fnc: theano.Function
        (batch_size, seq_len) -> (batch_size, seq_len, word_dim)

    vocab: Vocabulary
        Vocabulary instance

    Returns
    -------
        Python function: (batch_size, ) -> (batch_size, word_dim)
    """

    def _embedder(word_list):
        word_ids = vocab.encode(word_list)
        word_ids = np.array(word_ids)
        word_ids = word_ids.reshape((-1, 1)) # Just to adhere to theano.Function, whatever
        def_array, def_mask, def_map = retrieval.retrieve_and_pad(np.array(word_list).reshape(-1, 1))
        word_vectors = theano_fnc(word_ids, def_array, def_mask, def_map)
        word_vectors = word_vectors.reshape((len(word_list), -1))
        return word_vectors

    return _embedder

def construct_embedder(theano_fnc, vocab):
    """
    Parameters
    ----------
    theano_fnc: theano.Function
        (batch_size, seq_len) -> (batch_size, seq_len, word_dim)

    vocab: Vocabulary
        Vocabulary instance

    Returns
    -------
        Python function: (batch_size, ) -> (batch_size, word_dim)
    """

    def _embedder(word_list):
        word_ids = vocab.encode(word_list)
        word_ids = np.array(word_ids)
        word_ids = word_ids.reshape((-1, 1)) # Just to adhere to theano.Function, whatever
        word_vectors = theano_fnc(word_ids)
        word_vectors = word_vectors.reshape((len(word_list), -1))
        return word_vectors

    return _embedder


def evaluate_similarity(w, X, y, restrict_to_words=None):
    """
    Calculate Spearman correlation between cosine similarity of the model
    and human rated similarity of word pairs

    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.

    X: array, shape: (n_samples, 2)
      Word pairs

    y: vector, shape: (n_samples,)
      Human ratings

    Returns
    -------
    cor: float
      Spearman correlation
    """

    from web.embedding import Embedding

    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    mean_vector = np.mean(w.vectors, axis=0, keepdims=True)
    A = np.vstack(w.get(word, mean_vector) for word in X[:, 0])
    B = np.vstack(w.get(word, mean_vector) for word in X[:, 1])
    scores = np.array([v1.dot(v2.T) for v1, v2 in zip(A, B)])
    return scipy.stats.spearmanr(scores, y).correlation


class RetrievalPrintStats(SimpleExtension):
    """
    Prints statistics about Retrieval object
    """

    def __init__(self, retrieval, **kwargs):
        self._retrieval = retrieval
        super(RetrievalPrintStats, self).__init__(**kwargs)

    def __getstate__(self):
        dict_ = dict(self.__dict__)
        if '_retrieval' in dict_:
            del dict_['_retrieval']
        return dict_

    def add_records(self, log, record_tuples):
        """Helper function to add monitoring records to the log."""
        for name, value in record_tuples:
            if not name:
                raise ValueError("monitor variable without name")
            log.current_row[name] = value

    def do(self, *args, **kwargs):
        if self._retrieval is not None:
            d = self._retrieval._debug_info
            record_tuples = []
            record_tuples.append(
                ("retrieval_distinct_mis_ratio",
                 d['N_missed_distinct_words'] / max(1, float(d['N_distinct_words']))))
            record_tuples.append(("retrieval_N_words", d['N_words']))
            record_tuples.append(("retrieval_N_excluded_words", d['N_excluded_words']))
            record_tuples.append(("retrieval_N_distinct_words", d['N_distinct_words']))
            record_tuples.append(("retrieval_N_queried_words", d['N_queried_words']))
            record_tuples.append(("retrieval_mis_ratio",
                                  d['N_missed_words']
                                  /  max(1, float(d['N_queried_words']))))
            record_tuples.append(("retrieval_drop_def_ratio",
                                  d['N_dropped_def'] /  max(1, float(d['N_def']))))
            if len(d['missed_word_sample']) >= 20:
                record_tuples.append(("retrieval_missed_word_sample",
                    numpy.random.choice(d['missed_word_sample'], 20, replace=False)))
            self.add_records(self.main_loop.log, record_tuples)

class SimilarityWordEmbeddingEval(SimpleExtension):
    """
    Parameters
    ----------

    embedder: function: word -> vector
    """

    def __init__(self, embedder, prefix="", **kwargs):
        try:
            from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_RW
        except ImportError:
            raise RuntimeError("Please install web (https://github.com/kudkudak/word-embeddings-benchmarks)")

        self._embedder = embedder
        self._prefix = prefix

        # Define tasks
        logger.info("Downloading benchmark data")
        tasks = { # TODO: Pick a bit better tasks
            "MEN": fetch_MEN(),
            "WS353": fetch_WS353(),
            "SIMLEX999": fetch_SimLex999(),
            "RW": fetch_RW()
        }

        # Print sample data
        for name, data in iteritems(tasks):
            logger.info(
            "Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format(name, data.X[0][0], data.X[0][1],
                data.y[0]))

        logger.info("Checking embedder for " + prefix)
        logger.info(embedder(["love"])[0, 0:5]) # Test embedder

        self._tasks = tasks

        super(SimilarityWordEmbeddingEval, self).__init__(**kwargs)

    def __getstate__(self):
        dict_ = dict(self.__dict__)
        if '_embedder' in dict_:
            del dict_['_embedder']
        if '_tasks' in dict_:
            del dict_['_tasks']
        return dict_

    def add_records(self, log, record_tuples):
        """Helper function to add monitoring records to the log."""
        for name, value in record_tuples:
            if not name:
                raise ValueError("monitor variable without name")
            log.current_row[name] = value

    def do(self, *args, **kwargs):
        # Embedd
        all_words = []
        all_words_vectors = []

        # TODO: Do it at once?
        for task in self._tasks:
            for row in self._tasks[task].X:
                for w in row:
                    all_words.append(w)
                    all_words_vectors.append(self._embedder([w]))
        W = dict(zip(np.array(all_words).reshape((-1,)), all_words_vectors))

        # Calculate results using helper function
        record_items = []
        for name, data in iteritems(self._tasks):
            eval = evaluate_similarity(W, data.X, data.y)
            record_items.append((self._prefix + "_" + name, eval))

        self.add_records(self.main_loop.log, record_items)


class DumpCSVSummaries(SimpleExtension):
    def __init__(self, save_path, **kwargs):
        self._save_path = save_path

        if not os.path.exists(os.path.join(self._save_path, "logs.csv")):
            self._current_log = defaultdict(list)
        else:
            self._current_log = pd.read_csv(os.path.join(self._save_path, "logs.csv"))
            self._current_log = {col: list(self._current_log[col].values) for col in self._current_log.columns}
            logging.warning("Loaded {} columns and {} rows from logs.csv".format(len(self._current_log), len(self._current_log.values()[0])))

        super(DumpCSVSummaries, self).__init__(**kwargs)

    def do(self, *args, **kwargs):
        for key, value in self.main_loop.log.current_row.items():
            try:
                float_value = float(value)
            except:
                continue

            if not key.startswith("val") and not key.startswith("train") and not key.startswith("test"):
                key = "train_" + key

            if key not in self._current_log:
                self._current_log[key] = []

            self._current_log[key].append(float_value)

        # Make sure all logs have same length (for csv serialization)
        max_len = max([len(v) for v in self._current_log.values()])
        for k in self._current_log:
            if len(self._current_log[k]) != max_len:
                self._current_log[k] += [self._current_log[k][-1] for _ in range(max_len - len(self._current_log[k]))]

        pd.DataFrame(self._current_log).to_csv(os.path.join(self._save_path, "logs.csv"))



class LoadNoUnpickling(Load):
    """Like `Load` but without unpickling.

    Avoids unpiclkling the main loop by assuming that the log
    and the iteration state were saved separately.

    """

    def load_to(self, main_loop):
        with open(self.path, "rb") as source:
            main_loop.model.set_parameter_values(load_parameters(source))
            if self.load_iteration_state:
                main_loop.iteration_state = load(source, name='iteration_state')
            if self.load_log:
                main_loop.log = load(source, name='log')

class PrintMessage(SimpleExtension):
    """Prints log messages to the screen."""
    def __init__(self, msg, **kwargs):
        self._msg = msg
        kwargs.setdefault("before_first_epoch", True)
        kwargs.setdefault("on_resumption", True)
        kwargs.setdefault("after_training", True)
        kwargs.setdefault("after_epoch", True)
        kwargs.setdefault("on_interrupt", True)
        super(PrintMessage, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        print(self._msg)


class StartFuelServer(SimpleExtension):

    def __init__(self, stream, stream_path, script_path="start_fuel_server.py", hwm=100, *args, **kwargs):
        self._stream = stream
        self._hwm = hwm
        self._stream_path = stream_path
        self._script_path = script_path
        super(StartFuelServer, self).__init__(*args, **kwargs)

    def do(self, *args, **kwars):
        with open(self._stream_path, 'w') as dst:
            cPickle.dump(self._stream, dst, 0)
        port = get_free_port()
        self.main_loop.data_stream.port = port
        logger.debug("Starting the Fuel server on port " + str(port))
        ret = subprocess.Popen(
            [self._script_path,
                self._stream_path, str(port), str(self._hwm)])
        time.sleep(0.1)
        if ret.returncode is not None:
            raise Exception()
        atexit.register(lambda: os.kill(ret.pid, signal.SIGINT))

class IntermediateCheckpoint(Checkpoint):
    """
    Allows checkpointing intermediate models every_n_batches or
    every_n_epochs
    """
    def __init__(self, *args, **kwargs):
        super(IntermediateCheckpoint, self).__init__(*args, **kwargs)
        self.base_path = self.path
        # every epoch or batch
        if len(self._conditions) != 1:
            raise ValueError("too many conditions")
        condition = self._conditions[0]
        name, predicate, args = condition
        if (not predicate) or (not predicate.num):
            raise ValueError("wrong condition is not every n ...")
        self.every_n = predicate.num

    def do(self, *args, **kwargs):
        iterations_done = self.main_loop.log.status['iterations_done']
        self.path = "{}.after_batch_{}.tar".format(self.base_path, iterations_done)
        super(IntermediateCheckpoint, self).do(*args, **kwargs)


class CGStatistics(SimpleExtension):

    def __init__(self, **kwargs):
        kwargs.setdefault('before_first_epoch', True)
        kwargs.setdefault('on_resumption', True)
        super(CGStatistics, self).__init__(**kwargs)

    def do(self, *args, **kwargs):
        logger.info("Computation graph statistics:")
        cost_cg = ComputationGraph(self.main_loop.algorithm.cost)
        updates_cg = ComputationGraph(
            [u[1] for u in self.main_loop.algorithm.updates
             if isinstance(u[1], theano.Variable)])
        cost_nodes = io_toposort(cost_cg.inputs, cost_cg.outputs)
        updates_nodes = io_toposort(updates_cg.inputs, updates_cg.outputs)

        cost_scan_nodes = [
            node for node in cost_nodes
            if isinstance(node.op, Scan)]
        updates_scan_nodes = [
            node for node in updates_nodes
            if isinstance(node.op, Scan)]
        final_scan_nodes = [
            node for node in self.main_loop.algorithm._function.maker.fgraph.apply_nodes
            if isinstance(node.op, Scan)]

        logger.info("SCAN NODES IN THE COST GRAPH:")
        for n in cost_scan_nodes:
            logger.info(n.op.name)
        logger.info("SCAN NODES IN THE UPDATES GRAPH:")
        for n in updates_scan_nodes:
            logger.info(n.op.name)
        logger.info("SCAN NODES IN THE FINAL GRAPH:")
        for n in final_scan_nodes:
            logger.info(n.op.name)
