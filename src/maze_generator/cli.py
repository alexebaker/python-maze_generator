# -*- coding: utf-8 -*-

import argparse

from maze_generator import __version__
from maze_generator import Maze


def get_parser():
    parser = argparse.ArgumentParser(
        description='Generates a maze.')

    parser.add_argument('--version', action='version', version='%(prog)s v' + __version__)

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
    return parser


def run():
    """Parses cli arguments and runs the relevant functions."""
    cli_args = vars(get_parser().parse_args())
    maze = Maze(**cli_args)
    maze.generate()
    maze.find_solution()
    maze.display()
    return


if __name__ == '__main__':
    run()
