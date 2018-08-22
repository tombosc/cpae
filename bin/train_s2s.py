#!/usr/bin/env python

from dictlearn.def_autoencoder_training import train_model
from dictlearn.s2s_configs import configs_ae
from dictlearn.main import main


if __name__ == "__main__":
    main(configs_ae, train_model)
