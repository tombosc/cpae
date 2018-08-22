"""Config registry.

Config registry is just a dictionary of config, but it copies
the configs on read and on write to avoid subtle bugs.

"""

import copy

class ConfigRegistry(object):

    def __init__(self):
        self._configs = {}

    def get_root_config(self):
        return self['root']

    def set_root_config(self, config):
        self['root'] = config

    def __getitem__(self, name):
        # copy on read
        return copy.deepcopy(self._configs[name])

    def __setitem__(self, name, config):
        if name in self._configs:
            raise KeyError("Config already registered " + name)
        self._configs[name] = copy.deepcopy(config)

    def keys(self):
        return self._configs.keys()

