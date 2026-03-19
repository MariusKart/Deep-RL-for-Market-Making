
import torch
import matplotlib.pyplot as plt
from core.SimulationEnvironment import *
from core.models import *
from config.constants import *
from scipy.interpolate import interp1d
from torch import optim
import itertools

device_used = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)    
np.random.seed(42)
from pathlib import Path


# Produce a 1D Grid of Q and the corresponding value functions in the separable (uncorelated) case given "myopic" policy

def solve_1d_value_fixed_policy(
    market: Market, i,
    LB_i, UB_i,
    rfq_size, 
    delta_bid,
    delta_ask,
):
    grid = np.arange(LB_i, UB_i, rfq_size, dtype=float)
    n = grid.size

    p_bid = float(market.f(i, delta_bid))
    p_ask = float(market.f(i, delta_ask))

    gamma_rl = (2.0 * market.lambda_RFQs[i]) / (RF + 2.0 * market.lambda_RFQs[i])

    def psi(q):
        return 0.5 * GAMMA * np.sqrt(market.Sigma[i][i] * q * q)

    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    for k, q in enumerate(grid):
        can_up = (k + 1 < n)   # q + Δ, bid trade
        can_dn = (k - 1 >= 0)  # q - Δ, ask trade

        # coefficient of theta(q)
        coeff_kk = 1.0

        rhs = 0.0

        # Bid side: prob 1/2
        if can_up:
            # trade
            coeff_kk -= 0.5 * (1.0 - p_bid) * gamma_rl
            A[k, k + 1] += -0.5 * p_bid * gamma_rl

            rhs += 0.5 * p_bid * (rfq_size * delta_bid)
            rhs += 0.5 * p_bid * (-psi(q + rfq_size) / (RF + 2.0 * market.lambda_RFQs[i]))
            rhs += 0.5 * (1.0 - p_bid) * (-psi(q) / (RF + 2.0 * market.lambda_RFQs[i]))
        else:
            # at upper boundary, bid RFQ arrives but no inventory increase allowed
            coeff_kk -= 0.5 * gamma_rl
            rhs += 0.5 * (-psi(q) / (RF + 2.0 * market.lambda_RFQs[i]))

        # Ask side: prob 1/2
        if can_dn:
            coeff_kk -= 0.5 * (1.0 - p_ask) * gamma_rl
            A[k, k - 1] += -0.5 * p_ask * gamma_rl

            rhs += 0.5 * p_ask * (rfq_size * delta_ask)
            rhs += 0.5 * p_ask * (-psi(q - rfq_size) / (RF + 2.0 * market.lambda_RFQs[i]))
            rhs += 0.5 * (1.0 - p_ask) * (-psi(q) / (RF + 2.0 * market.lambda_RFQs[i]))
        else:
            # at lower boundary, ask RFQ arrives but no inventory decrease allowed
            coeff_kk -= 0.5 * gamma_rl
            rhs += 0.5 * (-psi(q) / (RF + 2.0 * market.lambda_RFQs[i]))

        A[k, k] = coeff_kk
        b[k] = rhs

    V = np.linalg.solve(A, b)
    V = V - V[n // 2]
    return grid, V
def produce_initial_value_grid(market: Market, init_strategy, lb_risk, ub_risk, nb_bonds, sizes):
    grids = []
    Vs = []

    for i in range(nb_bonds):
        grid_i, V_i = solve_1d_value_fixed_policy(
            market=market,
            i=i,
            LB_i=float(lb_risk[i]),
            UB_i=float(ub_risk[i]),
            rfq_size=sizes[i],
            delta_bid=float(init_strategy[i]),
            delta_ask=float(init_strategy[i]),
        )
        grids.append(grid_i)
        Vs.append(V_i)
    return grids, Vs


def pretrain_critic(
    critic,
    grids,          # list of 1D grids, one per bond
    Vs,             # list of 1D value arrays, one per bond
    sizes,
    device="cpu",
    n_dense=100,
    lr=1e-2,
    max_steps=3000,
    tol=1e-4,
):
    nb_bonds = len(grids)

    # 1D interpolator for each bond
    interpolants = []
    dense_axes = []

    for i in range(nb_bonds):
        grid_i = np.asarray(grids[i], dtype=np.float32).reshape(-1)
        V_i = np.asarray(Vs[i], dtype=np.float32).reshape(-1)

        f_i = interp1d(grid_i, V_i, kind="cubic")
        interpolants.append(f_i)

        dense_axis_i = np.linspace(grid_i.min(), grid_i.max(), n_dense, dtype=np.float32)
        dense_axes.append(dense_axis_i)

    # Cartesian product grid in ORIGINAL q-space
    mesh = np.meshgrid(*dense_axes, indexing="ij")
    X_orig = np.stack([m.ravel() for m in mesh], axis=1).astype(np.float32)

    # Target = sum of 1D interpolated values
    Y_dense = np.zeros((X_orig.shape[0], 1), dtype=np.float32)
    for i in range(nb_bonds):
        Y_dense[:, 0] += interpolants[i](X_orig[:, i]).astype(np.float32)

    # critic ALWAYS receives q / sizes
    X_scaled = X_orig / np.asarray(sizes, dtype=np.float32).reshape(1, -1)

    X = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    Y = torch.tensor(Y_dense, dtype=torch.float32, device=device)

    critic = critic.to(device).float()
    critic.train()

    opt = optim.Adam(critic.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for step in range(max_steps):
        opt.zero_grad()
        pred = critic(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()


        if loss.item() < tol:
            print(f"stopped early at step {step}, loss={loss.item():.6f}")
            break

    return critic


def save_pretrained_critic(critic, selected):
    save_dir = Path("pretrained_critic")
    save_dir.mkdir(exist_ok=True)
    torch.save(critic.state_dict(), save_dir / f"critic_bonds_{selected}_{len(selected)}_bond_scenario.pt")
    print(f"Critic saved to : 'pretrained_critic/critic_bonds_{selected}_{len(selected)}_bond_scenario.pt'")
