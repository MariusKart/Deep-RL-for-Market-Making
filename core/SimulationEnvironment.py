from scipy.special import ndtr, ndtri
import numpy as np
from config.constants import *


class RFQ:
    def __init__(self, direction, size, i):
        self.direction = int(direction)  # 0/1
        self.size = float(size)
        self.i = int(i)


class Market:
    """
    lambdas_RFQ: array of shape (d,)
    Sigma: matrix (d,d)
    mu: (d,)
    f_parameters: (alpha, beta) where each is array shape (d,)
    """
    def __init__(self, lambdas_RFQ, Sigma, rf, sizes, f_parameters, seed=42):
        self.lambda_RFQs = np.asarray(lambdas_RFQ, dtype=float) 
        self.Sigma = np.asarray(Sigma, dtype=float)             
        self.rf = float(rf)           
        self.f_parameters = f_parameters                        
        self.rng = np.random.default_rng(seed)
        self.sizes= sizes

    def generateRFQs(self, n, seed=None):
        """
        RFQ generation: returns arrays (i, direction, size).
        """
        total_intensity = float(np.sum(self.lambda_RFQs)) 
        p_i = self.lambda_RFQs / np.sum(total_intensity)
        rng = self.rng if seed is None else np.random.default_rng(seed)
        direction = rng.integers(0, 2, size=n, endpoint=False).astype(np.int64)
        i = rng.choice(np.arange(len(self.lambda_RFQs)), size=n, p=p_i).astype(np.int64)
        size = self.sizes[i].astype(float)
        return i, direction, size
    

    def f(self, i, delta):
        """
            f(delta) = 1 - Phi(alpha + beta * asinh((delta - mu[i]) / Sigma[i,i]))
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
        Inverse mapping p -> delta :
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

    

    
class MarketMaker(): 
    def __init__(self, risk_aversion, inventory, UB_risk, LB_risk, Market, cash =0., PnL = 0.):
        self.inventory = inventory
        self.risk_aversion = risk_aversion
        self.UB_risk = UB_risk
        self.LB_risk = LB_risk
        self.cash = cash
        self.PnL = 0
        self.Market = Market

    def update_batch(self, inventories, Sigma, r, i, direction, delta, avg_sizes):
        """
        inventories: (B, d)
        i:           (B,)
        direction:   (B,)   1=bid RFQ -> you buy -> inventory increases
                            0=ask RFQ -> you sell -> inventory decreases
        delta:       (B,)
        avg_sizes (d,)

        Returns
        -------
        rewards_exp:      (B,)
        inventories_next: (B, d)
        executed:         (B,)
        p_eff:            (B,)
        
        """
        inv = np.asarray(inventories, dtype=float)
        
        if inv.ndim == 1:
            inv = inv.reshape(-1, 1)   # (B,) -> (B,1)
        inv = inv.copy()

        i = np.asarray(i, dtype=int).reshape(-1)
        direction = np.asarray(direction, dtype=int).reshape(-1)
        delta = np.asarray(delta, dtype=float).reshape(-1)
        delta = np.clip(delta, a_min= NU, a_max = 1e12)
        B, d = inv.shape
        if i.shape[0] != B or direction.shape[0] != B or delta.shape[0] != B:
            raise ValueError("inventories, i, direction, delta must have compatible batch sizes")

        lambdas = np.asarray(self.Market.lambda_RFQs, dtype=float)
        Lambda = 2.0 * float(lambdas.sum())
        denom = float(r) + Lambda

        # Base trade probability from quote
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

        # Raw executed branch
        inv_exec_raw = inv.copy()
        inv_exec_raw[np.arange(B), i] += dq

        # Feasibility: if the executed branch breaches limits, execution intensity is zero
        lower_ok = inv_exec_raw[np.arange(B), i] >= np.asarray(self.LB_risk, dtype=float)[i]
        upper_ok = inv_exec_raw[np.arange(B), i] <= np.asarray(self.UB_risk, dtype=float)[i]
        feasible = lower_ok & upper_ok

        # Effective trade probability
        p_eff = p * feasible.astype(float)

        # For infeasible trades, executed branch is never reached, so keep a safe placeholder
        inv_exec = inv_exec_raw.copy()
        inv_exec[~feasible] = inv[~feasible]

        psi_q_exec = 0.5 * float(GAMMA) * np.sqrt(np.einsum("bi,ij,bj->b", inv_exec, Sigma, inv_exec))

        # Expected immediate reward with effective probability
        rewards_exp = p_eff * size * delta - (p_eff * psi_q_exec + (1.0 - p_eff) * psi_q) / denom

        # Simulated execution for rollout propagation
        executed = (np.random.rand(B) < p_eff)

        inv_next = inv.copy()
        idx = np.where(executed)[0]
        if idx.size > 0:
            inv_next[idx, i[idx]] += dq[idx]

        return rewards_exp, inv_next, executed, p_eff