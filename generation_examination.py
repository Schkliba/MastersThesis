#!/usr/bin/env python3

import argparse

# Replace this import with the actual module
from generation_examination_module import make_gen_examination


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
        "--type_out",
        "-t",
        type=str,
        default="base",
        help="type of output"
    )

    parser.add_argument(
        "--gen_start",
        "-l",
        type=int,
        default=0,
        help="type of output"
    )
    parser.add_argument(
        "--gen_end",
        "-u",
        type=int,
        default=0,
        help="type of output"
    )
    parser.add_argument(
        "--gen_step",
        "-p",
        type=int,
        default=1,
        help="type of output"
    )

    parser.add_argument(
        "--seeds",
        "-s",
        type=int,
        default=5,
        help="Output path for generated results"
    )

    parser.add_argument(
        "--source",
        "-R",
        default="optuna",
        help="Source of choice"
    )

    args = parser.parse_args()
    seeds =list(range(101, 101+args.seeds))
    generations = list(range(args.gen_start, args.gen_end, args.gen_step))

    print(seeds)
    make_gen_examination(
        en=args.environment,
        alg=args.algorithm,
        seeds=seeds,
        source=args.source,
        type_out=args.type_out,
        generations=generations,
        container=args.container,
        name=args.name
    )


if __name__ == "__main__":
    main()