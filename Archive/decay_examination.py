#!/usr/bin/env python3

import argparse

# Replace this import with the actual module
from decay_grid_search import make_decay_examination


def main():
    parser = argparse.ArgumentParser(
        description="Run final examination generation."
    )

    parser.add_argument(
        "--environment",
        "-e",
        type=str,
        required=True,
        help="Environment name"
    )

    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        required=True,
        help="Algorithm name"
    )

    parser.add_argument(
        "--container",
        "-c",
        type=str,
        required=True,
        help="Container name"
    )

    parser.add_argument(
        "--name",
        "-o",
        type=str,
        required=True,
        help="Output path for generated results"
    )
    parser.add_argument(
        "--seeds",
        "-s",
        type=int,
        default=3,
        help="Output path for generated results"
    )

    parser.add_argument(
        "--source",
        "-R",
        default="optuna",
        help="Source of choice"
    )

    args = parser.parse_args()
    seeds =list(range(121, 121+args.seeds))
    print(seeds)
    make_decay_examination(
        en=args.environment,
        alg=args.algorithm,
        seeds=seeds,
        source=args.source,
        container=args.container,
        name=args.name
    )


if __name__ == "__main__":
    main()