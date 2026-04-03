from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from otc_mm.plotting import plot_two_bond_surfaces


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 3D surfaces for a 2-bond trained model: "
            "optimal probabilities, optimal quotes, and critic value."
        )
    )
    parser.add_argument(
        "--bonds",
        type=int,
        nargs=2,
        required=True,
        help="Exactly 2 bond indices in [0, 19]",
    )
    parser.add_argument("--methodology", type=str, default="classic", choices=["classic", "greedy"])
    parser.add_argument(
        "--width_in_sizes",
        type=int,
        default=5,
        help="Plot q_j on the grid {-k*s_j, ..., +k*s_j}",
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Do not export the surface dataframes to CSV",
    )
    parser.add_argument(
        "--map_location",
        type=str,
        default="cpu",
        help="Torch map_location for loading checkpoints",
    )

    args = parser.parse_args()

    bonds = list(args.bonds)
    if any((b < 0 or b > 19) for b in bonds):
        raise ValueError("Each bond index must be between 0 and 19 included")
    if bonds[0] == bonds[1]:
        raise ValueError("The 2 bonds must be distinct")
    if args.width_in_sizes < 1:
        raise ValueError("--width_in_sizes must be >= 1")

    out = plot_two_bond_surfaces(
        selected_bonds=bonds,
        methodology=args.methodology,
        width_in_sizes=args.width_in_sizes,
        save_csv=not args.no_csv,
        map_location=args.map_location,
    )

    print(f"Saved plots to: {out['out_dir']}")


if __name__ == "__main__":
    main()