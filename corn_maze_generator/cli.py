import argparse

from .field import Field


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generates a maze.')

    parser.add_argument(
        '--density', '-d',
        type=float,
        default=0.2,
        help='Density of the maze.')

    parser.add_argument(
        '--complexity', '-c',
        type=float,
        default=0.8,
        help='Complexity of the maze')

    parser.add_argument(
        '--diameter',
        type=int,
        default=8,
        help='Size of the isle in the maze.')

    parser.add_argument(
        '--size',
        type=int,
        default=(4, 2),
        nargs=2,
        help='Dimensions of the maze.')

    parser.set_defaults()
    return parser.parse_args()


def run():
    """Parses cli arguments and runs the requested functions."""
    cli_args = parse_args()
    field = Field(**cli_args)
    field.generate()
    field.find_solution()
    field.display()
    return
