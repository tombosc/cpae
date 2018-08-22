"""
Few utilities
"""

import os, sys, logging
from contextlib import contextmanager
from logging import handlers
import datetime
from collections import OrderedDict
import socket
import numpy
from itertools import product
import pickle

def softmax(v, T):
    exp_v = numpy.exp(v/T)
    return exp_v / numpy.sum(exp_v)


def vec2str(vector):
    """Transforms a fixed size vector into a unicode string."""
    return u"".join(map(unichr, vector)).strip('\00')


def str2vec(str_, length):
    """Trasforms a string into a fixed size numpy.array

    Adds padding, if necessary. Truncates, if necessary.

    Importanty, if the input is a unicode string, the resulting
    array with contain unicode codes.

    """
    vector = numpy.array(map(ord, str_))[:length]
    pad_length = max(0, length - len(str_))
    return numpy.pad(vector, (0, pad_length), 'constant')


def rename(var, name):
    var.name = name
    return var


def smart_sum(x):
    for i in range(x.ndim):
        x = x.sum(axis=-1)
    return x


def masked_root_mean_square(x, mask):
    """Masked root mean square for a 3D tensor"""
    return (smart_sum((x * mask[:, :, None]) ** 2) / x.shape[2] / mask.sum()) ** 0.5


def get_free_port():
    # Copy-paste from
    # http://stackoverflow.com/questions/2838244/get-open-tcp-port-in-python
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port




def kwargs_namer(**fnc_kwargs):
    return "_".join("{}={}".format(k, v) for k, v in OrderedDict(**fnc_kwargs).iteritems() if k not in ['run_name'])

def utc_timestamp():
    return str(int(10 * (datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds()))


## Config logger

def parse_logging_level(logging_level):
    """
    :param logging_level: Logging level as string
    :return: Parsed logging level
    """
    lowercase = logging_level.lower()
    if lowercase == 'debug': return logging.DEBUG
    if lowercase == 'info': return logging.INFO
    if lowercase == 'warning': return logging.WARNING
    if lowercase == 'error': return logging.ERROR
    if lowercase == 'critical': return logging.CRITICAL
    raise ValueError('Logging level {} could not be parsed.'.format(logging_level))


def configure_logger(name = __name__,
                     console_logging_level = logging.INFO,
                     file_logging_level = logging.INFO,
                     log_file = None,
                     redirect_stdout=False,
                     redirect_stderr=False):
    """
    Configures logger
    :param name: logger name (default=module name, __name__)
    :param console_logging_level: level of logging to console (stdout), None = no logging
    :param file_logging_level: level of logging to log file, None = no logging
    :param log_file: path to log file (required if file_logging_level not None)
    :return instance of Logger class
    """
    if console_logging_level is None and file_logging_level is None:
        return # no logging

    if isinstance(console_logging_level, (str, unicode)):
        console_logging_level = parse_logging_level(console_logging_level)

    if isinstance(file_logging_level, (str, unicode)):
        file_logging_level = parse_logging_level(file_logging_level)

    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if console_logging_level is not None:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(format)
        ch.setLevel(console_logging_level)
        logger.addHandler(ch)

    if file_logging_level is not None:
        if log_file is None:
            raise ValueError("If file logging enabled, log_file path is required")
        fh = handlers.RotatingFileHandler(log_file, maxBytes=(1048576*5), backupCount=7)
        fh.setFormatter(format)
        logger.addHandler(fh)

    logger.info("Logging configured!")

    if redirect_stderr:
        sys.stderr = LoggerWriter(logger.warning)
    if redirect_stdout:
        sys.stdout = LoggerWriter(logger.info)

    return logger

def copy_streams_to_file(log_file, stdout=True, stderr=True):
    logger = logging.getLogger("_copy_stdout_stderr_to_file")
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    format = logging.Formatter("%(message)s")
    fh = handlers.RotatingFileHandler(log_file, maxBytes=(1048576 * 5), backupCount=7)
    fh.setFormatter(format)
    logger.addHandler(fh)

    if stderr:
        sys.stderr = LoggerWriter(logger.warning)

    if stdout:
        sys.stdout = LoggerWriter(logger.info)


class LoggerWriter:
    """
    This class can be used when we want to redirect stdout or stderr to a logger instance.
    Example of usage:

    log = logging.getLogger('foobar')
    sys.stdout = LoggerWriter(log.debug)
    sys.stderr = LoggerWriter(log.warning)
    """
    def __init__(self, level, also_print=False):
        self.level = level
        self.also_print = also_print

    def write(self, message):
        if message != '\n':
            self.level(message)
        if self.also_print:
            print(message)

    def flush(self):
        pass


## Dima's redirection code

class Fork(object):
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    def write(self, data):
        self.file1.write(data)
        self.file2.write(data)

    def flush(self):
        self.file1.flush()
        self.file2.flush()


@contextmanager
def replace_logging_stream(file_):
    root = logging.getLogger()
    if len(root.handlers) != 1:
        raise ValueError("Don't know what to do with many handlers")
    if not isinstance(root.handlers[0], logging.StreamHandler):
        raise ValueError
    stream = root.handlers[0].stream
    root.handlers[0].stream = file_
    try:
        yield
    finally:
        root.handlers[0].stream = stream


@contextmanager
def replace_standard_stream(stream_name, file_):
    stream = getattr(sys, stream_name)
    setattr(sys, stream_name, file_)
    try:
        yield
    finally:
        setattr(sys, stream_name, stream)


def run_with_redirection(stdout_path, stderr_path, func):
    def func_wrapper(*args, **kwargs):
        with open(stdout_path, 'a', 1) as out_dst:
            with open(stderr_path, 'a', 1) as err_dst:
                out_fork = Fork(sys.stdout, out_dst)
                err_fork = Fork(sys.stderr, err_dst)
                with replace_standard_stream('stderr', err_fork):
                    with replace_standard_stream('stdout', out_fork):
                        with replace_logging_stream(err_fork):
                            func(*args, **kwargs)
    return func_wrapper

class IterDictValues(dict):
    """
    special dict with iterator that iterates on key and element of values
    for example, {k: [v1, v2]} will iterate over [(k, v1), (k, v2)]
    """
    def __iter__(self):
        self._iter_k = self.iterkeys()
        self._cur_k = None
        self._prod = None
        return self

    def next(self):
        if self._prod:
            try:
                k, v, n = next(self._prod)
                return (k,v,n)
            except:
                self._prod = None

        if not self._prod:
            self._cur_k = next(self._iter_k)
            self._len = len(self[self._cur_k])
            self._prod = product([self._cur_k], self[self._cur_k], [self._len])
        try:
            k, v, n = next(self._prod)
            return (k,v,n)
        except:
            self._prod = None


def dict_diff(dict1, dict2):
    for key in set(dict1.keys()).union(set(dict2.keys())):
        if not key in dict2:
            print '<', key, ':', dict1[key]
        elif not key in dict1:
            print '>', key, ':', dict2[key]
        elif dict1[key] != dict2[key]:
            print '<', key, ':', dict1[key]
            print '>', key, ':', dict2[key]

def serialize_embeddings(embeddings, format_, dest_dir, fname):
    """
    Serialize embeddings into file fname in dest_dir using format_

    embeddings: dict mapping entries to np.array or list of np.array
    dest_dir: destination dir where the embedding file will be written
    format_: either 'dict' or 'glove'
    fname: filename (without the extension)
    """
    if format_ == 'dict':
        with open(os.path.join(dest_dir, fname + ".pkl"), "wb") as f:
            pickle.dump(embeddings, f)
    elif format_ == 'glove':
        with open(os.path.join(dest_dir, fname + ".txt"), "wb") as f:
            for k, embs in embeddings.iteritems():
                if type(embs) == list:
                    for emb in embs:
                        f.write(k + ' ' + ' '.join([str(e) for e in emb]) + '\n')
                else:
                    f.write(k + ' ' + ' '.join([str(e) for e in embs]) + '\n')


