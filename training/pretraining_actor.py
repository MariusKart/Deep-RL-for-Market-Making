
import torch
from core.SimulationEnvironment import *
from core.models import *
from config.constants import *
from scipy.optimize import minimize
import torch.optim as optim
from pathlib import Path
torch.manual_seed(42)    
np.random.seed(42)

device_used = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Produce the myopic probabilities and quotes as a target for the actors : 

def myopic_probs(selected_bonds: int, market_env: Market):
    """
    For each bond i, compute myopic delta_i* = argmax_{delta>=0} delta * f_i(delta),
    then return p_i = f_i(delta_i*).
    """
    p = np.zeros(len(selected_bonds), dtype=float)
    delta_star = np.zeros(len(selected_bonds), dtype=float)

    for i in range(len(selected_bonds)):
        def objective(x):
            delta = x[0]
            
            return -(delta * market_env.f(selected_bonds[i],delta))

        res = minimize(
                objective,
                x0=[1.0],
                bounds=[(0.005, 1e12)],
            )
        delta_i = float(res.x[0])
        delta_star[i] = delta_i

        p[i] = float(market_env.f(selected_bonds[i], delta_i))

    return p, delta_star


def pretrain_actor(
    actor,
    target_p_i,
    LB_risk,
    UB_risk,
    avg_sizes,
    batch_size=50,
    epochs=1000,
    lr=1e-3,
    device=device_used,
):
    d = LB_risk.shape[0]

    LB = torch.tensor(LB_risk, dtype=torch.float32, device=device).reshape(1, d)
    UB = torch.tensor(UB_risk, dtype=torch.float32, device=device).reshape(1, d)
    avg = torch.tensor(avg_sizes, dtype=torch.float32, device=device).reshape(1, d)


    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(actor.parameters(), lr=lr)

    actor = actor.to(device)
    actor.train()

    for _ in range(epochs):
        optimizer.zero_grad()

        q_rand = LB + (UB - LB) * torch.rand(batch_size, d, device=device)
        q_rand = q_rand / avg

        pred = actor(q_rand)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)

        target = torch.full((batch_size, 1), float(target_p_i), dtype=torch.float32, device=device)

        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

    return actor


def save_pretrained_actors(actors, selected):
    save_dir = Path("pretrained_actors")
    save_dir.mkdir(exist_ok=True)

    for i, actor in enumerate(actors):
        torch.save(actor.state_dict(), save_dir / f"actor_{selected[i]}_{len(selected)}_scenario.pt")
        print(f"Actor saved to : 'actor_{selected[i]}_{len(selected)}_scenario.pt'")