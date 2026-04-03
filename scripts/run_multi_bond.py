from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from otc_mm.experiments import run_multi_bond_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-bond experiment. "
            "Missing single-bond learned datasets are generated automatically first."
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

    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--nb_steps", type=int, default=500)
    parser.add_argument("--long_horizon", type=int, default=10_000)
    parser.add_argument("--nb_short_rollouts", type=int, default=100)
    parser.add_argument("--short_horizon", type=int, default=100)

    parser.add_argument("--critic_batch_size", type=int, default=50)
    parser.add_argument("--actor_batch_size", type=int, default=70)
    parser.add_argument("--n_epochs_critic", type=int, default=1)
    parser.add_argument("--n_epochs_actor", type=int, default=1)
    parser.add_argument("--update_risk_after", type=int, default=50)

    parser.add_argument("--critic_lr", type=float, default=1e-2)
    parser.add_argument("--actor_lr", type=float, default=1e-2)

    parser.add_argument("--dataset_pretrain_samples", type=int, default=20000)
    parser.add_argument("--dataset_pretrain_actor_lr", type=float, default=1e-3)
    parser.add_argument("--dataset_pretrain_actor_epochs", type=int, default=1000)
    parser.add_argument("--dataset_pretrain_actor_batch_size", type=int, default=256)
    parser.add_argument("--dataset_pretrain_critic_lr", type=float, default=1e-3)
    parser.add_argument("--dataset_pretrain_critic_max_steps", type=int, default=3000)
    parser.add_argument("--dataset_pretrain_critic_tol", type=float, default=1e-4)

    parser.add_argument("--lb_init", type=float, nargs="+", default=None)
    parser.add_argument("--ub_init", type=float, nargs="+", default=None)
    parser.add_argument("--lb_final", type=float, nargs="+", default=None)
    parser.add_argument("--ub_final", type=float, nargs="+", default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    bonds = list(dict.fromkeys(args.bonds))
    if len(bonds) == 0:
        raise ValueError("--bonds cannot be empty")
    if any((b < 0 or b > 19) for b in bonds):
        raise ValueError("Each bond index must be between 0 and 19 included")

    d = len(bonds)

    def _maybe_vector(x, name):
        if x is None:
            return None
        if len(x) != d:
            raise ValueError(f"{name} must contain exactly {d} values")
        return x

    result = run_multi_bond_experiment(
        selected_bonds=bonds,
        methodology=args.methodology,
        hidden_dim=args.hidden_dim,
        nb_steps=args.nb_steps,
        long_horizon=args.long_horizon,
        nb_short_rollouts=args.nb_short_rollouts,
        short_horizon=args.short_horizon,
        critic_batch_size=args.critic_batch_size,
        actor_batch_size=args.actor_batch_size,
        n_epochs_critic=args.n_epochs_critic,
        n_epochs_actor=args.n_epochs_actor,
        update_risk_after=args.update_risk_after,
        critic_lr=args.critic_lr,
        actor_lr=args.actor_lr,
        dataset_pretrain_samples=args.dataset_pretrain_samples,
        dataset_pretrain_actor_lr=args.dataset_pretrain_actor_lr,
        dataset_pretrain_actor_epochs=args.dataset_pretrain_actor_epochs,
        dataset_pretrain_actor_batch_size=args.dataset_pretrain_actor_batch_size,
        dataset_pretrain_critic_lr=args.dataset_pretrain_critic_lr,
        dataset_pretrain_critic_max_steps=args.dataset_pretrain_critic_max_steps,
        dataset_pretrain_critic_tol=args.dataset_pretrain_critic_tol,
        lb_init=_maybe_vector(args.lb_init, "--lb_init"),
        ub_init=_maybe_vector(args.ub_init, "--ub_init"),
        lb_final=_maybe_vector(args.lb_final, "--lb_final"),
        ub_final=_maybe_vector(args.ub_final, "--ub_final"),
        seed=args.seed,
        device=args.device,
    )

    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()