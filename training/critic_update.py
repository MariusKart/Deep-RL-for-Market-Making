import numpy as np
from config.constants import *
from core.SimulationEnvironment import *
from core.models import *
import torch
torch.manual_seed(42)    
np.random.seed(42)

def update_critic_td(
    critic : Critic,
    target_critic: Critic,
    optimizer,
    S,
    A,  
    R,
    S_next,
    market,
    avg_sizes,
    batch_size=50,
    n_epochs=1,
    r_bar=None,
    device="cpu",
    
):


    critic = critic.to(device)
    target_critic = target_critic.to(device)

    S_np  = np.asarray(S, dtype=np.float32)
    Sp_np = np.asarray(S_next, dtype=np.float32)
    A_np  = np.asarray(A, dtype=np.float32)
    R_np  = np.asarray(R, dtype=np.float32).reshape(-1)

    N = S_np.shape[0]
    if N == 0:
        return 0.0, (0.0 if r_bar is None else float(r_bar))

    assert Sp_np.shape == S_np.shape, f"S_next must match S, got {Sp_np.shape} vs {S_np.shape}"
    assert A_np.shape[0] == N and A_np.shape[1] >= 2, f"A must be (N,2+) with [i,a], got {A_np.shape}"
    assert R_np.shape[0] == N, f"R must be length N, got {R_np.shape[0]} vs {N}"

    if r_bar is None:
        r_bar = float(R_np.mean())

    # ----- center reward -----
    R_center = (R_np - float(r_bar)).astype(np.float32)

    # ----- Obtain p_t from a_t then apply inv_f to obtain quote_t -----

    bond_ids = A_np[:, 0].astype(np.int64)
    a_raw = A_np[:, 1].astype(np.float32)
    a_clip = np.clip(a_raw, NU, 1.0 - NU)

    size_vec = np.asarray(avg_sizes, dtype=np.float32)  # (d,)

    deltas = np.empty(N, dtype=np.float32)
    for k in range(N):
        i = int(bond_ids[k])
        deltas[k] = float(market.inv_f(i, float(a_clip[k])))

    p = np.empty(N, dtype=np.float32)
    for i in np.unique(bond_ids):
        mask = (bond_ids == i)
        p[mask] = np.asarray(market.f(int(i), deltas[mask]), dtype=np.float32)

    p = np.clip(p, NU, 1.0 - NU)


    denom_sizes = size_vec.reshape(1, -1)  # (1,d)

    S_t = torch.as_tensor(S_np / denom_sizes, dtype=torch.float32, device=device)
    Sp_t = torch.as_tensor(Sp_np / denom_sizes, dtype=torch.float32, device=device)
    r_t = torch.as_tensor(R_center, dtype=torch.float32, device=device).unsqueeze(1)
    p_t = torch.as_tensor(p, dtype=torch.float32, device=device).unsqueeze(1)

    critic.train()
    target_critic.eval()

    total_loss, n_batches = 0.0, 0

    for _ in range(n_epochs):
        perm = torch.randperm(N, device=device)

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]

            s  = S_t[idx]
            sp = Sp_t[idx]
            r  = r_t[idx]
            pb = p_t[idx]

            with torch.no_grad():
                V_trade = target_critic(sp)
                V_stay  = target_critic(s)

                if V_trade.dim() == 1:
                    V_trade = V_trade.unsqueeze(1)
                if V_stay.dim() == 1:
                    V_stay = V_stay.unsqueeze(1)

                y = r + DISCOUNT_RL * (pb * V_trade + (1.0 - pb) * V_stay)

            v = critic(s)
            if v.dim() == 1:
                v = v.unsqueeze(1)

            loss = torch.mean((v - y) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()
            n_batches += 1

    return total_loss / max(n_batches, 1), float(r_bar)