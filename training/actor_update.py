import numpy as np
import torch
from config.constants import *
from core.SimulationEnvironment import *
from core.models import *

torch.manual_seed(42)
np.random.seed(42)

def update_actor_i(
    actor_i,
    critic,
    optimizer,
    market,
    S_i,
    D_i,
    i,
    Sigma,
    lb,
    ub,
    avg_sizes,
    batch_size=2048,
    n_epochs=1,
    eps_half_width=0.05,
    max_grad_norm=1.0,
    device=None,
):
    if device is None:
        try:
            device = next(actor_i.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    actor_i = actor_i.to(device)
    critic = critic.to(device)

    S_i = np.asarray(S_i, dtype=np.float32)
    D_i = np.asarray(D_i, dtype=np.int64).reshape(-1)

    n = S_i.shape[0]
    if n == 0:
        return {"loss": 0.0, "n": 0}

    avg_sizes_vec = np.asarray(avg_sizes, dtype=np.float32).reshape(-1)
    size_i = float(avg_sizes_vec[i])

    sign_np = np.where(D_i == 1, +1.0, -1.0).astype(np.float32)

    S_trade_i = S_i.copy()
    S_trade_i[:, i] += sign_np * size_i
    S_trade_i[:, i] = np.clip(S_trade_i[:, i], float(lb[i]), float(ub[i]))

    lambdas = np.asarray(market.lambda_RFQs, dtype=np.float32).reshape(-1)
    Lambda = 2.0 * float(lambdas.sum())
    denom = float(RF) + Lambda
    gamma_rl = Lambda / denom
    grad_scaler = 1.0 / (((2.0 * eps_half_width) ** 2) / 12.0)

    S_t = torch.as_tensor(S_i, dtype=torch.float32, device=device)
    S_trade_t = torch.as_tensor(S_trade_i, dtype=torch.float32, device=device)
    sign_t = torch.as_tensor(sign_np, dtype=torch.float32, device=device)

    avg_sizes_t = torch.as_tensor(avg_sizes_vec, dtype=torch.float32, device=device)
    Sigma_t = torch.as_tensor(np.asarray(Sigma, dtype=np.float32), dtype=torch.float32, device=device)

    denom_t = torch.tensor(denom, dtype=torch.float32, device=device)
    gamma_t = torch.tensor(gamma_rl, dtype=torch.float32, device=device)

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

        psi_q = 0.5 * float(GAMMA) * torch.sqrt(torch.einsum("bi,ij,bj->b", S_t, Sigma_t, S_t))
        psi_trade = 0.5 * float(GAMMA) * torch.sqrt(torch.einsum("bi,ij,bj->b", S_trade_t, Sigma_t, S_trade_t))

        X_glob = (S_t * sign_t.unsqueeze(1)) / avg_sizes_t
        p_curr_glob = actor_i(X_glob)
        if p_curr_glob.dim() == 2:
            p_curr_glob = p_curr_glob[:, 0]

        eps_glob = (torch.rand(n, device=device) * (2.0 * eps_half_width)) - eps_half_width
        p_eps_glob = torch.clamp(p_curr_glob + eps_glob, NU, 1.0 - NU)

        delta_curr_glob = torch.as_tensor(
            np.asarray(market.inv_f(i, p_curr_glob.detach().cpu().numpy()), dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        delta_eps_glob = torch.as_tensor(
            np.asarray(market.inv_f(i, p_eps_glob.detach().cpu().numpy()), dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )

        v_eps_glob = (
            p_eps_glob * (size_i * delta_eps_glob - psi_trade / denom_t + gamma_t * V_trade)
            + (1.0 - p_eps_glob) * (-psi_q / denom_t + gamma_t * V_q)
        )
        v_curr_glob = (
            p_curr_glob * (size_i * delta_curr_glob - psi_trade / denom_t + gamma_t * V_trade)
            + (1.0 - p_curr_glob) * (-psi_q / denom_t + gamma_t * V_q)
        )

        global_adv_std = (v_eps_glob - v_curr_glob).std() + 1e-8

    actor_i.train()

    total_loss = 0.0
    nb = 0

    for _ in range(n_epochs):
        perm = torch.randperm(n, device=device)

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]

            X_b = (S_t[idx] * sign_t[idx].unsqueeze(1)) / avg_sizes_t

            with torch.no_grad():
                p_curr = actor_i(X_b)
                if p_curr.dim() == 2:
                    p_curr = p_curr[:, 0]

                eps = (torch.rand(idx.shape[0], device=device) * (2.0 * eps_half_width)) - eps_half_width
                p_eps = torch.clamp(p_curr + eps, NU, 1.0 - NU)

                delta_curr = torch.as_tensor(
                    np.asarray(market.inv_f(i, p_curr.detach().cpu().numpy()), dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                )
                delta_eps = torch.as_tensor(
                    np.asarray(market.inv_f(i, p_eps.detach().cpu().numpy()), dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                )

                v_eps = (
                    p_eps * (size_i * delta_eps - psi_trade[idx] / denom_t + gamma_t * V_trade[idx])
                    + (1.0 - p_eps) * (-psi_q[idx] / denom_t + gamma_t * V_q[idx])
                )
                v_curr = (
                    p_curr * (size_i * delta_curr - psi_trade[idx] / denom_t + gamma_t * V_trade[idx])
                    + (1.0 - p_curr) * (-psi_q[idx] / denom_t + gamma_t * V_q[idx])
                )

                dv = (v_eps - v_curr) / global_adv_std
                dp = p_eps - p_curr

            p_for_grad = actor_i(X_b)
            if p_for_grad.dim() == 2:
                p_for_grad = p_for_grad[:, 0]

            loss = -(p_for_grad * dv * dp).mean() * grad_scaler

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_i.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            total_loss += float(loss.detach().item())
            nb += 1

    return {
        "loss": total_loss / max(nb, 1),
        "n": int(n),
        "global_adv_std": float(global_adv_std.detach().item()),
    }