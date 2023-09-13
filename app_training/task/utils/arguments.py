import argparse

import json

import os

import sys


def get_training_args():
    '''Parses the args for training.'''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--channel',
        required=True,
        type=str,
        help='channel'
    )

    parser.add_argument(
        '--model_type',
        required=True,
        type=str,
        help='model_type'
    )

    parser.add_argument(
        '--nr_of_epochs',
        required=True,
        type=int,
        help='nr_of_epochs'
    )

    try:
        args = parser.parse_args()
    except:  # noqa: E722
        parser.print_help()
        sys.exit(0)

    return args


def get_data_format_config(folder_path):
    '''returns dictionary of dataformat config from datasets folder.'''
    file_path = os.path.join(folder_path, 'config.json')
    with open(file_path, 'r') as j:
        contents = json.loads(j.read())

    return contents
