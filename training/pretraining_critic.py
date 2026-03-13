
import torch
import matplotlib.pyplot as plt
from core.SimulationEnvironment import *
from core.models import *
from config.constants import *
from scipy.optimize import minimize
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset
device_used = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

from pathlib import Path


# Produce a 1D Grid of Q and the corresponding value functions in the separable (uncorelated) case given "myopic" policy

def solve_1d_value_fixed_policy(
    market: Market, i,
    LB_i, UB_i,
    rfq_size, 
    delta_bid,
    delta_ask,
):
    grid = np.arange(LB_i, UB_i + 1e-12, rfq_size, dtype=float)
    n = grid.size

    p_bid = float(market.f(i, delta_bid))
    p_ask = float(market.f(i, delta_ask))

    gamma_rl = (2.0 * market.lambda_RFQs[i]) / (RF + 2.0 * market.lambda_RFQs[i])

    def psi(q):
        return 0.5 * GAMMA * market.Sigma[i][i] * q * q

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
    return grid, V


def produce_initial_value_grid(market :Market, init_strategy, lb_risk, ub_risk, nb_bonds, sizes):
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
        
    critic_target = np.sum(np.array(Vs), axis = 0)
    critic_input = np.array(grids).T
    return critic_target, critic_input


def pretrain_critic(
    critic: nn.Module,
    input, target,
    epochs = 100,
    lr = 5e-8,
    device= device_used,
):

    # NN input uses normalized inventory q/z
    X = torch.tensor(input, dtype=torch.float32)
    Y = torch.tensor(target, dtype=torch.float32).unsqueeze(1)


    critic = critic.to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    critic.train()
    for _ in range(epochs):
        xb = X.to(device)
        yb = Y.to(device)

        pred = critic(xb)        
        loss = loss_fn(pred, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return critic

def save_critic(critic, selected):
    save_dir = Path("pretrained_critic")
    save_dir.mkdir(exist_ok=True)
    torch.save(critic.state_dict(), save_dir / f"critic_bonds_{selected}_{len(selected)}_bond_scenario.pt")
    print(f"Critic saved to : 'pretrained_critic/critic_bonds_{selected}_{len(selected)}_bond_scenario.pt'")
