#!/usr/bin/env python3

import argparse
from adaptive_gridsearch_module import adaptive_grid_search




def main():
    parser = argparse.ArgumentParser(
        description="Run adaptive grid search."
    )

    parser.add_argument(
        "--environment",
        "-e",
        required=True,
        help="Environment name"
    )

    parser.add_argument(
        "--algorithm",
        "-a",
        required=True,
        help="Algorithm name"
    )

    parser.add_argument(
        "--run-name",
        "-r",
        required=True,
        help="Name of the run/experiment"
    )

    parser.add_argument(
        "--container",
        "-c",
        required=True,
        help="Container name"
    )

    parser.add_argument(
        "--hops",
        "-H",
        type=int,
        default=3,
        help="Number of adaptive search hops (default: 3)"
    )

    parser.add_argument(
        "--out-path",
        "-o",
        default="./Data/grid_search",
        help="Output directory (default: ./Data/grid_search)"
    )

    parser.add_argument(
        "--source",
        "-S",
        default="optuna",
        help="source of choice"
    )

    args = parser.parse_args()

    adaptive_grid_search(
        en=args.environment,
        alg=args.algorithm,
        run_name=args.run_name,
        container=args.container,
        hops=args.hops,
        out_path=args.out_path,
    )


if __name__ == "__main__":
    main()