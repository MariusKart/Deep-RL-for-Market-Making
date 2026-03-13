
import torch
from core.SimulationEnvironment import *
from core.models import *
from config.constants import *
from scipy.optimize import minimize
import torch.optim as optim
from pathlib import Path


device_used = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Produce the myopic probabilities and quotes as a target for the actors : 

def myopic_probs(d: int, market_env: Market):
    """
    For each bond i, compute myopic delta_i* = argmax_{delta>=0} delta * f_i(delta),
    then return p_i = f_i(delta_i*).
    """
    p = np.zeros(d, dtype=float)
    delta_star = np.zeros(d, dtype=float)

    for i in range(d):
        def objective(x):
            delta = x[0]
            
            return -(delta * market_env.f(i,delta))

        res = minimize(
                objective,
                x0=[1.0],
                bounds=[(0.005, 1e12)],
            )
        delta_i = float(res.x[0])
        delta_star[i] = delta_i

        p[i] = float(market_env.f(i, delta_i))

    return p, delta_star


def pretrain_actor(actor, target_p_i, LB_risk, UB_risk, avg_sizes, n_samples=100000, epochs=100, lr=1e-2, device=device_used):
    d = LB_risk.shape[0]

    q = np.random.uniform(LB_risk, UB_risk, size=(n_samples, d)).astype(np.float32)
    q = q / avg_sizes.reshape(1, d).astype(np.float32)

    states = torch.tensor(q, dtype=torch.float32, device=device)           # (N,d)
    targets = torch.full((n_samples, 1), float(target_p_i), dtype=torch.float32, device=device)  # (N,1)

    optimizer = optim.Adam(actor.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    actor.train()
    for _ in range(epochs):
        pred = actor(states)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)  # (N,1)
        loss = loss_fn(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return actor


def save_actors(actors, selected):
    save_dir = Path("pretrained_actors")
    save_dir.mkdir(exist_ok=True)

    for i, actor in enumerate(actors):
        torch.save(actor.state_dict(), save_dir / f"actor_{selected[i]}_pretrained.pt")
        print(f"Actor saved to : 'actor_{selected[i]}_pretrained.pt'")