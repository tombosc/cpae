from blocks.bricks import Initializable

class Initializable2(Initializable):
    """Hack to add more initialization schemes."""
    def __init__(self, recurrent_weights_init=None,
                 embeddings_weights_init=None, **kwargs):
        super(Initializable2, self).__init__(**kwargs)
        self.recurrent_weights_init = recurrent_weights_init
        self.embeddings_weights_init = embeddings_weights_init

    def _push_initialization_config(self):
        super(Initializable2, self)._push_initialization_config()
        for child in self.children:
            if isinstance(child, Initializable2):
                child.recurrent_weights_init = self.recurrent_weights_init
                child.embeddings_weights_init = self.embeddings_weights_init
