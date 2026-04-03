from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from otc_mm.plotting import plot_single_bond_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the single-bond outputs already saved by training: "
            "optimal probabilities, optimal quotes, and training avg reward."
        )
    )
    parser.add_argument("--bond", type=int, required=True, help="Bond index in [0, 19]")
    parser.add_argument("--methodology", type=str, default="classic", choices=["classic", "greedy"])
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Do not export the plotting dataframes to CSV",
    )

    args = parser.parse_args()

    if not (0 <= args.bond <= 19):
        raise ValueError("--bond must be between 0 and 19 included")

    out = plot_single_bond_bundle(
        bond=args.bond,
        methodology=args.methodology,
        save_csv=not args.no_csv,
    )

    print(f"Saved plots to: {out['out_dir']}")


if __name__ == "__main__":
    main()