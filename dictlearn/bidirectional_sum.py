# -*- coding: utf-8 -*-
import copy

from picklable_itertools.extras import equizip
from theano import tensor

from blocks.bricks.base import application, lazy
from blocks.bricks.parallel import Fork
from blocks.bricks.simple import Initializable, Linear
from blocks.bricks.recurrent import BaseRecurrent, recurrent

class BidirectionalSum(Initializable):
    """Bidirectional network.
    A bidirectional network is a combination of forward and backward
    recurrent networks which process inputs in different order. Unlike 
    the regular Bidirectional, it sums the hidden states of the forward 
    and backward LSTMs instead of concatenating.

    Parameters
    ----------
    prototype : instance of :class:`BaseRecurrent`
        A prototype brick from which the forward and backward bricks are
        cloned.
    Notes
    -----
    See :class:`.Initializable` for initialization parameters.
    """
    has_bias = False

    @lazy()
    def __init__(self, prototype, **kwargs):
        self.prototype = prototype

        children = [copy.deepcopy(prototype) for _ in range(2)]
        children[0].name = 'forward'
        children[1].name = 'backward'
        kwargs.setdefault('children', []).extend(children)
        super(BidirectionalSum, self).__init__(**kwargs)

    @application
    def apply(self, *args, **kwargs):
        """Applies forward and backward networks and concatenates outputs."""
        forward_s, forward_c = self.children[0].apply(as_list=True, *args, **kwargs)
        backward_s, backward_c = [x[::-1] for x in
                    self.children[1].apply(reverse=True, as_list=True,
                                           *args, **kwargs)]
        return forward_s + backward_s, forward_c + backward_c

    @apply.delegate
    def apply_delegate(self):
        return self.children[0].apply

    def get_dim(self, name):
        if name in self.apply.outputs:
            return self.prototype.get_dim(name)
        return self.prototype.get_dim(name)
