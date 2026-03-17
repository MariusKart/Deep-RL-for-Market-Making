import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

def save_actors(actors, selected):
    save_dir = Path("trained_models")
    save_dir.mkdir(exist_ok=True)

    for i, actor in enumerate(actors):
        torch.save(actor.state_dict(), save_dir / f"actor_{selected[i]}_{len(selected)}_scenario.pt")
        print(f"Actor saved to : 'trained_models/actor_{selected[i]}_{len(selected)}_scenario.pt'")
        
def save_critic(critic, selected):
    save_dir = Path("trained_models")
    save_dir.mkdir(exist_ok=True)
    torch.save(critic.state_dict(), save_dir / f"critic_bonds_{selected}_{len(selected)}_bond_scenario.pt")
    print(f"Critic saved to : 'trained_models/critic_bonds_{selected}_{len(selected)}_bond_scenario.pt'")

def plot_avg_reward(avg_reward, save_dir="figures", filename="avg_reward.eps"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / filename

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.plot(
        avg_reward,
        linewidth=2.2,
        label="Average reward"
    )

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Average reward", fontsize=12)

    ax.grid(False, alpha=0.35)
    ax.margins(x=0.01)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(file_path, format="eps", bbox_inches="tight")
    plt.show()

    return file_path
def save_reward_to_csv(avg_reward, column_name, csv_path="data/training_reward.csv"):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame({column_name: avg_reward})

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df_existing[column_name] = df_new[column_name]
        df_existing.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)

    return csv_path

def plot_critic(
    critic,
    critic_input,   # ORIGINAL q
    sizes,
    device="cpu",
    n_plot=500,
):
    critic.eval()

    q_orig = np.asarray(critic_input, dtype=np.float32).reshape(-1, 1)

    # dense grid in ORIGINAL q-space for plotting
    q_plot_orig = np.linspace(
        q_orig[:, 0].min(),
        q_orig[:, 0].max(),
        n_plot,
        dtype=np.float32
    ).reshape(-1, 1)

    # critic receives q / SIZES
    q_plot_scaled = q_plot_orig / np.asarray(sizes, dtype=np.float32).reshape(1, -1)
    X_plot = torch.tensor(q_plot_scaled, dtype=torch.float32, device=device)

    with torch.no_grad():
        y_plot = critic(X_plot).cpu().numpy().reshape(-1)

    plt.figure(figsize=(7, 5))
    plt.plot(q_plot_orig[:, 0], y_plot, label="critic value")
    plt.xlabel("q")
    plt.ylabel("theta")
    plt.title("Critic value function")

    plt.grid(True)
    plt.show()
    
    
    

def plot_critic_2d(
    critic,
    LB_RISK,
    UB_RISK,
    sizes,
    selected_bonds,
    device="cpu",
):
    critic.eval()

    grid_0 = np.arange(
        LB_RISK[0],
        UB_RISK[0] + sizes[0],
        sizes[0],
        dtype=np.float32
    )

    grid_1 = np.arange(
        LB_RISK[1],
        UB_RISK[1] + sizes[1],
        sizes[1],
        dtype=np.float32
    )

    Q0, Q1 = np.meshgrid(grid_0, grid_1, indexing="ij")

    # original q grid, but critic gets q / sizes
    X = np.column_stack([
        Q0.ravel() / sizes[0],
        Q1.ravel() / sizes[1],
    ]).astype(np.float32)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        V = critic(X_t)
        if V.dim() == 2 and V.shape[1] == 1:
            V = V[:, 0]
        V = V.cpu().numpy()

    V_grid = V.reshape(Q0.shape)

    out_path = Path(f"figures/critic_value_function_{selected_bonds}.eps")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        Q0,
        Q1,
        V_grid,
        cmap="viridis",    
        linewidth=0,
        antialiased=True,
        shade=True
    )

    ax.set_xlabel(f"Inventory in bond {selected_bonds[0]}")
    ax.set_ylabel(f"Inventory in bond {selected_bonds[1]}")
    ax.set_zlabel("Value")
    ax.set_title("Value")


    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    fig.savefig(out_path, format="eps", bbox_inches="tight")
    plt.show()