from theano import tensor

from blocks.monitoring.aggregation import Mean

class Perplexity(Mean):

    def get_aggregator(self):
        aggregator = super(Perplexity, self).get_aggregator()
        if aggregator.readout_variable.ndim > 0:
            raise ValueError("can only compute perplexity for a scalar")
        aggregator.readout_variable = tensor.exp(aggregator.readout_variable)
        return aggregator

