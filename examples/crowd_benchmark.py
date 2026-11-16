"""Public-data test of reliability learning (design: research/10_public_data_benchmark.md).

Usage::

    python -m examples.crowd_benchmark DATA_DIR --download   # fetch and verify (once)
    python -m examples.crowd_benchmark DATA_DIR [--orders 10] [--trust-points 101]
        [--trust-mean 0.6667] [--trust-concentration 3] [--datasets snow2008_rte ...]

Prints one JSON document with the results for every data set.
"""

from __future__ import annotations

import argparse
import json
import sys

from kanm.crowd_data import DATASETS, download, load
from kanm.reliability import run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir")
    parser.add_argument("--download", action="store_true", help="download and verify the files")
    parser.add_argument("--orders", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-points", type=int, default=101,
                        help="discrete reliability-grid resolution (higher is slower)")
    parser.add_argument("--trust-mean", type=float, default=2.0 / 3.0)
    parser.add_argument("--trust-concentration", type=float, default=3.0)
    parser.add_argument("--min-gold-reports", type=int, default=20)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    args = parser.parse_args(argv)
    if args.download:
        download(args.data_dir, tuple(args.datasets))
    report = {
        name: run_benchmark(
            load(name, args.data_dir), orders=args.orders, seed=args.seed,
            trust_points=args.trust_points, trust_mean=args.trust_mean,
            trust_concentration=args.trust_concentration,
            min_gold_reports=args.min_gold_reports,
        )
        for name in args.datasets
    }
    json.dump(report, sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
