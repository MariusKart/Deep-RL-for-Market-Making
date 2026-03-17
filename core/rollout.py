from core.SimulationEnvironment import *
from config.constants import *
import torch



def rollout(policy, starting_inventory, horizon, market, r, Sigma, lb, ub, avg_sizes):
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
                avg_sizes=avg_sizes
            )

            inventories = np.asarray(inv_next, dtype=np.float32)

            Abase_out[t, :, 0] = i_t.astype(np.float32)
            Abase_out[t, :, 1] = a_base
            D_out[t] = dir_t
            R_out[t] = np.asarray(r_batch, dtype=np.float32)

    S_flat = S_out.reshape(N, d)
    Strade_flat = Strade_out.reshape(N, d)
    Abase_flat = Abase_out.reshape(N, 2)
    D_flat = D_out.reshape(N)
    R_flat = R_out.reshape(N)

    if single:
        return (
            S_flat[:T],
            Abase_flat[:T],
            D_flat[:T],
            R_flat[:T],
            Strade_flat[:T],
        )

    return S_flat, Abase_flat, D_flat, R_flat, Strade_flat