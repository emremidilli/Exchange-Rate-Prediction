import argparse

import sys


def get_training_args():
    '''
    Parses the args for training.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_type',
        required=False,
        default='nbeats',
        type=str,
        help='model_type'
    )

    parser.add_argument(
        '--nr_of_epochs',
        required=False,
        default=3,
        type=int,
        help='nr_of_epochs'
    )

    try:
        args = parser.parse_args()
    except:  # noqa: E722
        parser.print_help()
        sys.exit(0)

    return args
