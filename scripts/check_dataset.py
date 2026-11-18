"""Check a consolidated dataset and write summary.json into its directory.

Example:
    python scripts/check_dataset.py --data dataset

Exits with status 1 if any check fails (e.g. too few cases solved cleanly).
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfd_drag.check import check_dir  # noqa: E402
from cfd_drag.params import CHECK  # noqa: E402
from cfd_drag.paths import DEFAULT_DATASET_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--min-ok", type=float, default=CHECK.min_ok_fraction)
    args = ap.parse_args()
    summary, problems = check_dir(args.data, replace(CHECK, min_ok_fraction=args.min_ok))
    brief = {k: summary.get(k) for k in ["n", "n_ok", "ok_fraction", "failures", "shapes", "regimes"]}
    print(json.dumps(brief, indent=2))
    for p in problems:
        print("FAIL:", p)
    print("PASSED" if not problems else "FAILED", f"(summary: {args.data / 'summary.json'})")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
