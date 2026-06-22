#!/usr/bin/env python3

import argparse

# Replace this import with the actual module
from final_examination_module import make_final_examination


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
        default=30,
        help="Output path for generated results"
    )

    parser.add_argument(
        "--source",
        "-R",
        default="list",
        help="Source of choice"
    )

    args = parser.parse_args()
    seeds =list(range(101, 101+args.seeds))
    make_final_examination(
        en=args.environment,
        alg=args.algorithm,
        seeds=seeds,
        container=args.container,
        name=args.name,
        source = args.source
    )


if __name__ == "__main__":
    main()