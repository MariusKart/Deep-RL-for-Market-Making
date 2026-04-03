from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri

from .constants import GAMMA, NU


class RFQ:
    def __init__(self, direction, size, i):
        self.direction = int(direction)
        self.size = float(size)
        self.i = int(i)


class Market:
    """
    lambdas_RFQ: array of shape (d,)
    Sigma: matrix (d,d)
    f_parameters: (alpha, beta, mu, scale)
    """

    def __init__(self, lambdas_RFQ, Sigma, rf, sizes, f_parameters, seed=42):
        self.lambda_RFQs = np.asarray(lambdas_RFQ, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.rf = float(rf)
        self.f_parameters = f_parameters
        self.rng = np.random.default_rng(seed)
        self.sizes = np.asarray(sizes, dtype=float)

    def generateRFQs(self, n, seed=None):
        """
        RFQ generation: returns arrays (i, direction, size).
        """
        total_intensity = float(np.sum(self.lambda_RFQs))
        p_i = self.lambda_RFQs / total_intensity

        rng = self.rng if seed is None else np.random.default_rng(seed)
        direction = rng.integers(0, 2, size=n, endpoint=False).astype(np.int64)
        i = rng.choice(np.arange(len(self.lambda_RFQs)), size=n, p=p_i).astype(np.int64)
        size = self.sizes[i].astype(float)
        return i, direction, size

    def f(self, i, delta):
        """
        f(delta) = 1 - Phi(alpha + beta * asinh((delta - mu[i]) / scale[i]))
        """
        i = int(i)
        delta = np.asarray(delta)

        alpha, beta, mu, scale = self.f_parameters
        a = alpha[i]
        b = beta[i]
        m = mu[i]
        s = scale[i]

        z = a + b * np.arcsinh((delta - m) / s)
        return ndtr(-z)

    def inv_f(self, i, p):
        """
        Inverse mapping p -> delta:
            p = 1 - Phi(alpha + beta * asinh((delta - mu)/scale))
            delta = mu + scale * sinh((Phi^{-1}(1-p) - alpha)/beta)
        """
        i = int(i)
        p = np.clip(np.asarray(p), NU, 1.0 - NU)

        alpha, beta, mu, scale = self.f_parameters
        a = alpha[i]
        b = beta[i]
        m = mu[i]
        s = scale[i]

        z = ndtri(1.0 - p)
        return m + s * np.sinh((z - a) / b)


class MarketMaker:
    def __init__(self, risk_aversion, inventory, UB_risk, LB_risk, Market, cash=0.0, PnL=0.0):
        self.inventory = inventory
        self.risk_aversion = risk_aversion
        self.UB_risk = UB_risk
        self.LB_risk = LB_risk
        self.cash = cash
        self.PnL = PnL
        self.Market = Market

    def update_batch(self, inventories, Sigma, r, i, direction, delta, avg_sizes):
        """
        inventories: (B, d)
        i:           (B,)
        direction:   (B,)   1=bid RFQ -> you buy -> inventory increases
                            0=ask RFQ -> you sell -> inventory decreases
        delta:       (B,)
        avg_sizes:   (d,)

        Returns
        -------
        rewards_exp:      (B,)
        inventories_next: (B, d)
        executed:         (B,)
        p_eff:            (B,)
        """
        inv = np.asarray(inventories, dtype=float)

        if inv.ndim == 1:
            inv = inv.reshape(-1, 1)
        inv = inv.copy()

        i = np.asarray(i, dtype=int).reshape(-1)
        direction = np.asarray(direction, dtype=int).reshape(-1)
        delta = np.asarray(delta, dtype=float).reshape(-1)

        B, d = inv.shape
        if i.shape[0] != B or direction.shape[0] != B or delta.shape[0] != B:
            raise ValueError("inventories, i, direction, delta must have compatible batch sizes")

        lambdas = np.asarray(self.Market.lambda_RFQs, dtype=float)
        Lambda = 2.0 * float(lambdas.sum())
        denom = float(r) + Lambda

        p = np.empty(B, dtype=float)
        for j in np.unique(i):
            m = (i == j)
            p[m] = np.asarray(self.Market.f(int(j), delta[m]), dtype=float)
        p = np.clip(p, NU, 1.0 - NU)

        psi_q = 0.5 * float(GAMMA) * np.sqrt(np.einsum("bi,ij,bj->b", inv, Sigma, inv))

        avg_sizes = np.asarray(avg_sizes, dtype=float).reshape(-1)
        size = avg_sizes[i]
        sign = np.where(direction == 1, +1.0, -1.0)
        dq = sign * size

        inv_exec_raw = inv.copy()
        inv_exec_raw[np.arange(B), i] += dq

        lower_ok = inv_exec_raw[np.arange(B), i] >= np.asarray(self.LB_risk, dtype=float)[i]
        upper_ok = inv_exec_raw[np.arange(B), i] <= np.asarray(self.UB_risk, dtype=float)[i]
        feasible = lower_ok & upper_ok

        p_eff = p * feasible.astype(float)

        inv_exec = inv_exec_raw.copy()
        inv_exec[~feasible] = inv[~feasible]

        psi_q_exec = 0.5 * float(GAMMA) * np.sqrt(np.einsum("bi,ij,bj->b", inv_exec, Sigma, inv_exec))

        rewards_exp = p_eff * size * delta - (p_eff * psi_q_exec + (1.0 - p_eff) * psi_q) / denom

        executed = (np.random.rand(B) < p_eff)

        inv_next = inv.copy()
        idx = np.where(executed)[0]
        if idx.size > 0:
            inv_next[idx, i[idx]] += dq[idx]

        return rewards_exp, inv_next, executed, p_eff


def myopic_probs(selected_bonds, market_env: Market):
    """
    For each bond i, compute myopic delta_i* = argmax_{delta>=0} delta * f_i(delta),
    then return p_i = f_i(delta_i*).
    """
    p = np.zeros(len(selected_bonds), dtype=float)
    delta_star = np.zeros(len(selected_bonds), dtype=float)

    for i in range(len(selected_bonds)):
        def objective(x):
            delta = x[0]
            return -(delta * market_env.f(selected_bonds[i], delta))

        res = minimize(
            objective,
            x0=[1.0],
            bounds=[(0.005, 1e12)],
        )
        delta_i = float(res.x[0])
        delta_star[i] = delta_i
        p[i] = float(market_env.f(selected_bonds[i], delta_i))

    return p, delta_star


def rollout(policy, starting_inventory, horizon, market, r, Sigma, lb, ub, avg_sizes):
    """
    This function follows the simulation methodology described in the paper:
    draw a side, draw a bond, compute the policy and the resulting reward.

    The visited states (simulated) and rewards are stored. Additionally it stores
    the state of the inventory in case the trade was accepted, for convenience.
    """
    avg_sizes_vec = np.asarray(avg_sizes, dtype=np.float32)

    LBv = np.asarray(lb, dtype=np.float32)
    UBv = np.asarray(ub, dtype=np.float32)

    q0 = np.asarray(starting_inventory, dtype=np.float32)
    if q0.ndim == 1:
        inventories = q0.reshape(1, -1).copy()
        B = 1
        single = True
    elif q0.ndim == 2:
        inventories = q0.copy()
        B = inventories.shape[0]
        single = False
    else:
        raise ValueError(f"starting_inventory must be (d,) or (B,d), got {q0.shape}")

    d = inventories.shape[1]
    T = int(horizon)
    N = T * B

    mm = MarketMaker(GAMMA, np.zeros(d, dtype=float), UB_risk=UBv, LB_risk=LBv, Market=market)

    S_out = np.empty((T, B, d), dtype=np.float32)
    Strade_out = np.empty((T, B, d), dtype=np.float32)
    Abase_out = np.empty((T, B, 2), dtype=np.float32)
    D_out = np.empty((T, B), dtype=np.int64)
    R_out = np.empty((T, B), dtype=np.float32)

    try:
        device = next(policy[0].parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    for a in policy:
        a.eval()

    with torch.inference_mode():
        for t in range(T):
            q = inventories.copy()
            S_out[t] = q

            i_t = np.empty(B, dtype=np.int64)
            dir_t = np.empty(B, dtype=np.int64)
            sz_t = np.empty(B, dtype=np.float32)

            for b in range(B):
                i_one, dir_one, sz_one = market.generateRFQs(1)
                i_t[b] = int(np.asarray(i_one).reshape(-1)[0])
                dir_t[b] = int(np.asarray(dir_one).reshape(-1)[0])
                sz_t[b] = float(np.asarray(sz_one).reshape(-1)[0])

            q_norm_np = (q / avg_sizes_vec).astype(np.float32, copy=False)
            q_norm_t = torch.as_tensor(q_norm_np, dtype=torch.float32, device=device)

            a_base = np.empty(B, dtype=np.float32)
            uniq = np.unique(i_t)

            for j in uniq:
                m = (i_t == j)
                x = q_norm_t[m]

                side_sign_np = np.where(dir_t[m] == 1, 1.0, -1.0).astype(np.float32)
                side_sign_t = torch.as_tensor(side_sign_np, dtype=torch.float32, device=device).unsqueeze(1)
                x_actor = x * side_sign_t

                y = policy[int(j)](x_actor)
                if y.dim() == 2 and y.shape[1] == 1:
                    y = y[:, 0]
                elif y.dim() != 1:
                    raise ValueError(f"policy[{j}] must output (n,) or (n,1), got {tuple(y.shape)}")

                a_base[m] = y.detach().cpu().numpy().astype(np.float32)

            a_base = np.clip(a_base, NU, 1.0 - NU)

            delta_base = np.empty(B, dtype=np.float32)
            for j in uniq:
                m = (i_t == j)
                delta_base[m] = np.asarray(market.inv_f(int(j), a_base[m]), dtype=np.float32)

            trade_sign = np.where(dir_t == 1, +1.0, -1.0).astype(np.float32)
            q_exec = q.copy()
            q_exec[np.arange(B), i_t] += trade_sign * sz_t

            q_exec_stored = q_exec.copy()
            q_exec_stored = np.minimum(np.maximum(q_exec_stored, LBv), UBv)
            Strade_out[t] = q_exec_stored

            r_batch, inv_next, executed, p_eff = mm.update_batch(
                inventories=q,
                Sigma=Sigma,
                r=r,
                i=i_t,
                direction=dir_t,
                delta=delta_base,
                avg_sizes=avg_sizes,
            )

            inventories = np.asarray(inv_next, dtype=np.float32)

            Abase_out[t, :, 0] = i_t.astype(np.float32)
            Abase_out[t, :, 1] = a_base
            D_out[t] = dir_t
            R_out[t] = np.asarray(r_batch, dtype=np.float32)

    S_flat = S_out.reshape(N, d)
    Strade_flat = Strade_out.reshape(N, d)
    A_flat = Abase_out.reshape(N, 2)
    D_flat = D_out.reshape(N)
    R_flat = R_out.reshape(N)

    if single:
        return (
            S_flat[:T],
            A_flat[:T],
            D_flat[:T],
            R_flat[:T],
            Strade_flat[:T],
        )

    return S_flat, A_flat, D_flat, R_flat, Strade_flat