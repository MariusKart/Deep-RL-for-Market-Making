import numpy as np
import torch
from config.constants import *
from core.SimulationEnvironment import *
from core.models import *


def update_actor_i(
    actor_i,
    critic,
    optimizer,
    market,
    S_i,            # (N,d) only samples with RFQ bond == i
    p_base_i,       # (N,) base probabilities for bond i
    p_noisy_i,      # (N,) noisy probabilities for bond i
    D_i,            # (N,) 1 = bid RFQ -> MM buys -> inventory increases
    i,
    Sigma,
    lb, 
    ub,
    avg_sizes,
    batch_size=2048,
    n_epochs=1,
    device=None
):
    # ---- device ----
    if device is None:
        try:
            device = next(actor_i.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    actor_i = actor_i.to(device)
    critic = critic.to(device)

    # ---- numpy inputs ----
    S_i = np.asarray(S_i, dtype=np.float32)
    p_base_i = np.clip(np.asarray(p_base_i, dtype=np.float32).reshape(-1), NU, 1.0 - NU)
    p_noisy_i = np.clip(np.asarray(p_noisy_i, dtype=np.float32).reshape(-1), NU, 1.0 - NU)
    D_i = np.asarray(D_i, dtype=np.int64).reshape(-1)

    n = S_i.shape[0]
    if n == 0:
        return {"loss": 0.0, "n": 0}

    d = S_i.shape[1]
    avg_sizes_vec = np.asarray(avg_sizes, dtype=np.float32)
    size_i = float(avg_sizes_vec[i])

    # ---- q_trade ----
    # D_i == 1 => bid RFQ => MM buys => inventory increases
    sign = np.where(D_i == 1, +1.0, -1.0).astype(np.float32)

    S_trade_i = S_i.copy()
    S_trade_i[:, i] += sign * size_i
    S_trade_i[:, i] = np.clip(S_trade_i[:, i], float(lb[i]), float(ub[i]))

    # ---- market-side quote reconstruction ----
    delta_base_i = np.asarray(market.inv_f(i, p_base_i), dtype=np.float32)
    delta_noisy_i = np.asarray(market.inv_f(i, p_noisy_i), dtype=np.float32)

    dp_i = p_noisy_i - p_base_i

    # ---- constants ----
    lambdas = np.asarray(market.lambda_RFQs, dtype=np.float32).reshape(-1)
    Lambda = 2.0 * float(lambdas.sum())   # symmetric bid/ask intensity convention
    denom = float(RF) + Lambda
    gamma_rl = Lambda / denom

    # ---- torch tensors ----
    S_t = torch.as_tensor(S_i, dtype=torch.float32, device=device)
    S_trade_t = torch.as_tensor(S_trade_i, dtype=torch.float32, device=device)

    avg_sizes_t = torch.as_tensor(avg_sizes_vec, dtype=torch.float32, device=device)
    Sigma_t = torch.as_tensor(np.asarray(Sigma, dtype=np.float32), dtype=torch.float32, device=device)

    p_base_t = torch.as_tensor(p_base_i, dtype=torch.float32, device=device)
    p_noisy_t = torch.as_tensor(p_noisy_i, dtype=torch.float32, device=device)
    dp_t = torch.as_tensor(dp_i, dtype=torch.float32, device=device)

    delta_base_t = torch.as_tensor(delta_base_i, dtype=torch.float32, device=device)
    delta_noisy_t = torch.as_tensor(delta_noisy_i, dtype=torch.float32, device=device)

    denom_t = torch.tensor(denom, dtype=torch.float32, device=device)
    gamma_t = torch.tensor(gamma_rl, dtype=torch.float32, device=device)

    # ---- critic values + penalties ----
    with torch.inference_mode():
        critic.eval()

        S_norm_t = S_t / avg_sizes_t
        S_trade_norm_t = S_trade_t / avg_sizes_t

        V_q = critic(S_norm_t)
        V_trade = critic(S_trade_norm_t)

        if V_q.dim() == 2:
            V_q = V_q[:, 0]
        if V_trade.dim() == 2:
            V_trade = V_trade[:, 0]

        psi_q = 0.5 * float(GAMMA) * torch.einsum("bi,ij,bj->b", S_t, Sigma_t, S_t)
        psi_trade = 0.5 * float(GAMMA) * torch.einsum("bi,ij,bj->b", S_trade_t, Sigma_t, S_trade_t)

        score_base = (
            p_base_t * (size_i * delta_base_t - psi_trade / denom_t + gamma_t * V_trade)
            + (1.0 - p_base_t) * (-psi_q / denom_t + gamma_t * V_q)
        )

        score_noisy = (
            p_noisy_t * (size_i * delta_noisy_t - psi_trade / denom_t + gamma_t * V_trade)
            + (1.0 - p_noisy_t) * (-psi_q / denom_t + gamma_t * V_q)
        )

        Delta = score_noisy - score_base
        Delta_norm = (Delta) / (Delta.std() + 1e-8)

        # fixed finite-difference weights
        w = (Delta_norm * dp_t).detach()

    # ---- actor update ----
    actor_i.train()
    S_norm_actor = (S_t / avg_sizes_t).detach()

    total_loss = 0.0
    nb = 0

    for _ in range(n_epochs):
        perm = torch.randperm(n, device=device)

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]

            p_pred = actor_i(S_norm_actor[idx])
            if p_pred.dim() == 1:
                p_pred = p_pred.unsqueeze(1)
            elif p_pred.dim() == 2 and p_pred.shape[1] != 1:
                raise ValueError(f"actor_i must output (B,) or (B,1), got {tuple(p_pred.shape)}")

            loss = -(w[idx].unsqueeze(1) * p_pred).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().item())
            nb += 1

    return {
        "loss": total_loss / max(nb, 1),
        "n": int(n),
        "Delta_mean": float(Delta.mean().item()),
        "Delta_std": float(Delta.std().item()),
        "p_base_mean": float(p_base_t.mean().item()),
        "p_noisy_mean": float(p_noisy_t.mean().item()),
    }