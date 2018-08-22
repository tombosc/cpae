#!/usr/bin/env python
import os
import pprint
import argparse

import logging
logging.basicConfig(
    level='DEBUG',
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from dictlearn.util import run_with_redirection


def add_config_arguments(config, parser):
    for key, value in config.items():
        if isinstance(value, bool):
            parser.add_argument(
                "--" + key, dest=key, default=None, action="store_true",
                help="Enable a setting from the configuration")
            parser.add_argument(
                "--no_" + key, dest=key, default=None, action="store_false",
                help="Disable a setting from the configuration")
        else:
            convertor = type(value)
            # let's assume all the lists in our configurations will be
            # lists of ints
            if isinstance(value, list):
                convertor = lambda s: map(int, s.split(','))
            parser.add_argument(
                "--" + key, type=convertor,
                help="A setting from the configuration")


def main(config_registry, training_func, **training_func_kwargs):
    parser = argparse.ArgumentParser("Learning with a dictionary")
    parser.add_argument("--fast-start", action="store_true",
                        help="Start faster by skipping a few things at the start")
    parser.add_argument("--fuel-server", action="store_true",
                        help="Use standalone Fuel dataset server")
    parser.add_argument("--params",
                        help="Load parameters from a main loop")
    #parser.add_argument("--seed", type=int,
    #                    help="The random seed")
    parser.add_argument("config", help="The configuration")
    parser.add_argument("save_path", help="The destination for saving")
    add_config_arguments(config_registry.get_root_config(), parser)

    args = parser.parse_args()

    print args.seed
    # Modify the configuration with the command line arguments
    config = config_registry[args.config]
    for key in config:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)

    new_training_job = False
    if not os.path.exists(args.save_path):
        new_training_job = True
        os.mkdir(args.save_path)

    def call_training_func():
        pprint.pprint(config)
        if new_training_job:
            logger.info("Start a new job")
        else:
            logger.info("Continue an existing job")
        training_func(new_training_job, config, args.save_path,
                      args.params, args.fast_start, args.fuel_server, args.seed,
                      **training_func_kwargs)
    run_with_redirection(
        os.path.join(args.save_path, 'stdout.txt'),
        os.path.join(args.save_path, 'stderr.txt'),
        call_training_func)()


def main_evaluate(config_registry, evaluate_func):
    parser = argparse.ArgumentParser("Evaluation script")
    parser.add_argument("--part", help="Part")
    parser.add_argument("--dataset", help="Provide a dataset explicitly")
    parser.add_argument("--dest", help="Destination for outputs")
    parser.add_argument("--details", help="Output more details", action="store_true", default=False)
    parser.add_argument("--num-examples", type=int, help="Number of examples to read", default=-1)
    parser.add_argument("--qids", type=str, help="Comma-separate qids")
    parser.add_argument("config", help="The configuration")
    parser.add_argument("tar_path", help="The tar file with parameters")
    add_config_arguments(config_registry.get_root_config(), parser)

    args = parser.parse_args()

    # Modify the configuration with the command line arguments
    if args.config.endswith("json"):
        config = args.config
    else:
        config = config_registry[args.config]
        for key in config:
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        pprint.pprint(config)

    # For now this script just runs the language model training.
    # More stuff to come.
    kwargs = {}
    if args.qids:
        kwargs['qids'] = args.qids
    if args.dataset:
        kwargs['dataset'] = args.dataset
    if args.details:
        kwargs['details'] = True
    evaluate_func(config, args.tar_path, args.part, args.num_examples, args.dest, **kwargs)

def main_generate_embeddings(config_registry, generate_func):
    parser = argparse.ArgumentParser("Generate embeddings script")
    parser.add_argument("--part", default='train', help="Part")
    parser.add_argument("--dest", help="Destination for outputs", required=True)
    parser.add_argument("--format", dest='format_', type=str, help="format for embeddings: 'dict' or 'glove'")
    parser.add_argument("--encoder-embeddings", type=str, default=None,
                        help="Export encoder embeddings instead of definition: 'only', 'mixed', 'if_missing'")
    parser.add_argument("--average", action='store_true',
                        help="Average multi-prototype embeddings")
    parser.add_argument("config", help="The configuration")
    parser.add_argument("tar_path", help="The tar file with parameters")
    add_config_arguments(config_registry.get_root_config(), parser)

    args = parser.parse_args()

    # Modify the configuration with the command line arguments

    if args.config.endswith("json"):
        config = args.config
    else:
        config = config_registry[args.config]
        for key in config:
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        pprint.pprint(config)

    generate_func(config, args.tar_path, args.part, args.dest, args.format_,
                  args.average, args.encoder_embeddings)

def main_curriculum(config_registry, generate_func):
    parser = argparse.ArgumentParser("Generate embeddings script with curriculum")
    parser.add_argument("--dest", help="Destination for outputs", required=True)
    parser.add_argument("--format", dest='format_', type=str, help="format for embeddings: 'dict' or 'glove'")
    parser.add_argument("config", help="The configuration")
    parser.add_argument("tar_path", help="The tar file with parameters")
    parser.add_argument("file_list", help="The file with the list of data files")
    add_config_arguments(config_registry.get_root_config(), parser)

    args = parser.parse_args()

    # Modify the configuration with the command line arguments

    if args.config.endswith("json"):
        config = args.config
    else:
        config = config_registry[args.config]
        for key in config:
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        pprint.pprint(config)

    # For now this script just runs the language model training.
    with open(args.file_list, 'r') as f:
        files_to_read = f.read()
    file_list = files_to_read.split('\n')
    file_list = [os.path.join(os.path.dirname(files_to_read), fname) for fname in file_list]
    print file_list
    generate_func(config, args.tar_path, file_list, args.dest, args.format_)

