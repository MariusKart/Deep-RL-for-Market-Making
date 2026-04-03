from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from otc_mm.plotting import plot_multi_bond_learning_curve


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the multi-bond training avg reward as scatter + rolling average "
            "from the saved metrics."
        )
    )
    parser.add_argument(
        "--bonds",
        type=int,
        nargs="+",
        required=True,
        help="List of bond indices in [0, 19]",
    )
    parser.add_argument("--methodology", type=str, default="classic", choices=["classic", "greedy"])
    parser.add_argument(
        "--rolling_window",
        type=int,
        default=25,
        help="Rolling window for the smoothed avg reward line",
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Do not export the plotting dataframe to CSV",
    )

    args = parser.parse_args()

    bonds = list(dict.fromkeys(args.bonds))
    if len(bonds) == 0:
        raise ValueError("--bonds cannot be empty")
    if any((b < 0 or b > 19) for b in bonds):
        raise ValueError("Each bond index must be between 0 and 19 included")
    if args.rolling_window < 1:
        raise ValueError("--rolling_window must be >= 1")

    out = plot_multi_bond_learning_curve(
        selected_bonds=bonds,
        methodology=args.methodology,
        rolling_window=args.rolling_window,
        save_csv=not args.no_csv,
    )

    print(f"Saved plots to: {out['out_dir']}")


if __name__ == "__main__":
    main()