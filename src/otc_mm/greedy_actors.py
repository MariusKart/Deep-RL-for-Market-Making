from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

from .constants import GAMMA, NU, RF


class TableActor1D(nn.Module):
    """
    Greedy/table actor.

    The actor stores one probability per discrete inventory state for bond i.
    Its input is the normalized state x = q / avg_sizes, exactly like the MLP actor,
    but it only reads coordinate i and maps it back to the discrete inventory index.
    """

    def __init__(self, i, lb_i, ub_i, size_i, init_p=0.5):
        super().__init__()
        self.i = int(i)
        self.lb_i = float(lb_i)
        self.ub_i = float(ub_i)
        self.size_i = float(size_i)

        self.n_states = int(round((self.ub_i - self.lb_i) / self.size_i)) + 1
        self.register_buffer(
            "table",
            torch.full((self.n_states,), float(init_p), dtype=torch.float32),
        )

    def forward(self, x):
        """
        x shape: (batch, d), normalized by avg_sizes
        The actor reads x[:, i], converts back to signed inventory q_i,
        then indexes into the stored probability table.
        """
        xi = x[:, self.i]
        q_signed = xi * self.size_i

        idx = torch.round((q_signed - self.lb_i) / self.size_i).long()
        idx = torch.clamp(idx, 0, self.n_states - 1)

        return self.table[idx].unsqueeze(1)

    def set_table(self, values):
        v = torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self.table.device,
        ).reshape(self.n_states)
        self.table.copy_(v)

    def export_table(self):
        q_grid = self.lb_i + self.size_i * np.arange(self.n_states, dtype=np.float32)
        p_grid = self.table.detach().cpu().numpy().astype(np.float32)
        return q_grid, p_grid


def myopic_probs_local(market_env):
    """
    0..d-1 of the current selected basket.
    """
    d = len(market_env.lambda_RFQs)
    p = np.zeros(d, dtype=float)
    delta_star = np.zeros(d, dtype=float)

    for i in range(d):
        def objective(x):
            delta = x[0]
            return -(delta * market_env.f(i, delta))

        res = minimize(
            objective,
            x0=[1.0],
            bounds=[(0.005, 1e12)],
        )
        delta_i = float(res.x[0])
        delta_star[i] = delta_i
        p[i] = float(market_env.f(i, delta_i))

    return p, delta_star


def build_index_bounds(lb, ub, avg_sizes):
    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    avg_sizes = np.asarray(avg_sizes, dtype=float).reshape(-1)

    low_idx = np.rint(lb / avg_sizes).astype(np.int64)
    high_idx = np.rint(ub / avg_sizes).astype(np.int64)
    return low_idx, high_idx


def sample_discrete_starts(n_samples, low_idx, high_idx, avg_sizes, rng):
    d = len(low_idx)
    idx = np.empty((n_samples, d), dtype=np.int64)

    for j in range(d):
        idx[:, j] = rng.integers(low_idx[j], high_idx[j] + 1, size=n_samples)

    return idx.astype(np.float32) * np.asarray(avg_sizes, dtype=np.float32).reshape(1, -1)


def build_table_actors(sizes, lb_init, ub_init, market_simulator):
    """
    Build one greedy table actor per LOCAL bond index, initialized at the
    local myopic probability
    """
    p0, delta0 = myopic_probs_local(market_simulator)

    actors = [
        TableActor1D(
            i=j,
            lb_i=lb_init[j],
            ub_i=ub_init[j],
            size_i=sizes[j],
            init_p=float(p0[j]),
        )
        for j in range(len(sizes))
    ]

    return actors, p0, delta0


def greedy_refresh_actor_1d_from_critic(
    actor_i,
    critic,
    market,
    i,
    Sigma,
    avg_sizes,
    lb,
    ub,
    p_grid=None,
    r=RF,
    device="cpu",
):
    """
    Greedy policy improvement

    For each discrete state q_i in the actor table:
      - keep the other inventories at 0
      - evaluate the one-step value for all probabilities in p_grid
      - pick the probability maximizing the expected value
      - write it into the actor table
    """
    critic = critic.to(device)

    if p_grid is None:
        p_grid = torch.linspace(NU, 1.0 - NU, 101, device=device)
    else:
        p_grid = torch.as_tensor(p_grid, dtype=torch.float32, device=device)

    avg_sizes = np.asarray(avg_sizes, dtype=np.float32)
    size_i = float(avg_sizes[i])

    q_grid = actor_i.lb_i + actor_i.size_i * np.arange(actor_i.n_states, dtype=np.float32)
    n = actor_i.n_states
    d = len(avg_sizes)

    S = np.zeros((n, d), dtype=np.float32)
    S[:, i] = q_grid

    S_trade = S.copy()
    S_trade[:, i] = np.clip(S_trade[:, i] + size_i, actor_i.lb_i, actor_i.ub_i)

    S_t = torch.as_tensor(S / avg_sizes.reshape(1, -1), dtype=torch.float32, device=device)
    S_trade_t = torch.as_tensor(S_trade / avg_sizes.reshape(1, -1), dtype=torch.float32, device=device)

    S_raw_t = torch.as_tensor(S, dtype=torch.float32, device=device)
    S_trade_raw_t = torch.as_tensor(S_trade, dtype=torch.float32, device=device)
    Sigma_t = torch.as_tensor(np.asarray(Sigma, dtype=np.float32), dtype=torch.float32, device=device)

    lambdas = np.asarray(market.lambda_RFQs, dtype=np.float32).reshape(-1)
    Lambda = 2.0 * float(lambdas.sum())
    denom = float(r) + Lambda
    gamma_rl = Lambda / denom

    with torch.no_grad():
        V_q = critic(S_t)
        V_trade = critic(S_trade_t)

        if V_q.dim() == 2:
            V_q = V_q[:, 0]
        if V_trade.dim() == 2:
            V_trade = V_trade[:, 0]

        psi_q = 0.5 * float(GAMMA) * torch.sqrt(
            torch.einsum("bi,ij,bj->b", S_raw_t, Sigma_t, S_raw_t)
        )
        psi_trade = 0.5 * float(GAMMA) * torch.sqrt(
            torch.einsum("bi,ij,bj->b", S_trade_raw_t, Sigma_t, S_trade_raw_t)
        )

        delta_grid = torch.as_tensor(
            np.asarray(market.inv_f(i, p_grid.detach().cpu().numpy()), dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )

        trade_term = (
            size_i * delta_grid.unsqueeze(0)
            - psi_trade.unsqueeze(1) / denom
            + gamma_rl * V_trade.unsqueeze(1)
        )
        stay_term = (
            -psi_q.unsqueeze(1) / denom
            + gamma_rl * V_q.unsqueeze(1)
        )

        scores = p_grid.unsqueeze(0) * trade_term + (1.0 - p_grid.unsqueeze(0)) * stay_term
        best_idx = torch.argmax(scores, dim=1)
        p_star = p_grid[best_idx]

    actor_i.set_table(p_star)

    return {
        "q_grid": q_grid,
        "p_star": p_star.detach().cpu().numpy(),
    }