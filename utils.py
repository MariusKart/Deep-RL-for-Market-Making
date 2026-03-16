import torch
import matplotlib.pyplot as plt
from pathlib import Path
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