from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from otc_mm.experiments import run_single_bond_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single-bond experiment and save the learned 1D dataset."
    )

    parser.add_argument("--bond", type=int, required=True, help="Bond index in [0, 19]")
    parser.add_argument("--methodology", type=str, default="classic", choices=["classic", "greedy"])

    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--nb_steps", type=int, default=50)
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

    parser.add_argument("--pretrain_actor_lr", type=float, default=1e-3)
    parser.add_argument("--pretrain_actor_epochs", type=int, default=1000)

    parser.add_argument("--pretrain_critic_lr", type=float, default=1e-2)
    parser.add_argument("--pretrain_critic_max_steps", type=int, default=10000)
    parser.add_argument("--pretrain_critic_n_dense", type=int, default=5000)

    parser.add_argument("--lb_init", type=float, nargs="+", default=None)
    parser.add_argument("--ub_init", type=float, nargs="+", default=None)
    parser.add_argument("--lb_final", type=float, nargs="+", default=None)
    parser.add_argument("--ub_final", type=float, nargs="+", default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if not (0 <= args.bond <= 19):
        raise ValueError("--bond must be between 0 and 19 included")

    def _maybe_scalar_list(x):
        if x is None:
            return None
        if len(x) != 1:
            raise ValueError("For single-bond runs, bound arguments must contain exactly 1 value")
        return x

    result = run_single_bond_experiment(
        bond=args.bond,
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
        pretrain_actor_lr=args.pretrain_actor_lr,
        pretrain_actor_epochs=args.pretrain_actor_epochs,
        pretrain_critic_lr=args.pretrain_critic_lr,
        pretrain_critic_max_steps=args.pretrain_critic_max_steps,
        pretrain_critic_n_dense=args.pretrain_critic_n_dense,
        lb_init=_maybe_scalar_list(args.lb_init),
        ub_init=_maybe_scalar_list(args.ub_init),
        lb_final=_maybe_scalar_list(args.lb_final),
        ub_final=_maybe_scalar_list(args.ub_final),
        seed=args.seed,
        device=args.device,
    )

    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()