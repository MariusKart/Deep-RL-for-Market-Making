# Deep Reinforcement Learning for Market Making in Corporate Bonds

This repository contains a reproduction of the paper *Deep Reinforcement Learning for Market Making in Corporate Bonds* by Guéant and Manziuk, together with a simple policy-improvement variant developed as part of the Market Microstructure coursework in the MSc Mathematics and Finance at Imperial College London.

The project studies market making on multi-dealer-to-client corporate bond platforms. A dealer receives RFQs, chooses bid and ask quotes, earns spread when trades occur, and at the same time must control inventory risk. In low dimension, this problem can be approached with classical stochastic control and HJB methods. In higher dimension, these methods become difficult to use in practice because the inventory state space grows too quickly.

The main idea of the paper is to replace direct PDE-based resolution with a model-based reinforcement learning procedure. The market-making problem is written in discrete time at RFQ arrival times, and the quoting rule is learned with an actor--critic architecture. A key point is that the actor does not output quotes directly: it outputs probabilities of trade, which are then converted back into spreads through the execution model.

## What is implemented here

The repository reproduces the main reinforcement learning framework of the paper:
- a simulated market-making environment for correlated corporate bonds,
- inventory-based state dynamics,
- a critic network used to approximate the value function,
- an actor network used to parameterize the quoting policy through trade probabilities,
- training through rollouts, TD learning, and policy-improvement updates.

The code also includes an additional policy-improvement approach. Instead of learning the actor with random perturbations, the critic is kept fixed and the policy is improved by direct maximization of the one-step action-value over a discretized grid of admissible trade probabilities. This produces a simple greedy alternative that is easier to interpret in low dimension.

## Financial intuition

At the core of the problem is the usual market-making trade-off. Tight quotes increase the probability of trading and therefore the expected spread capture, but they also make inventory more volatile. Wide quotes reduce trading intensity and help control exposure. The optimal policy must therefore adapt to the current inventory and, in the multi-bond case, to the interaction between correlated positions.

In the one-bond case, the learned policy has the expected shape: when inventory is too short, bid quotes become more aggressive in order to buy back inventory; when inventory is already too long, bid quotes become less aggressive in order to avoid accumulating more risk. In the two-bond case, the learned surfaces reflect the fact that inventory in one bond can partly hedge exposure in another correlated bond.

## Results

The main reproduction results are the following.

In the one-bond case, the reinforcement learning procedure recovers the expected policy shape and reaches an average reward of about 200 per RFQ.

In the two-bond case, using bonds 1 and 6, the learned value function and quote surfaces reflect the correlation structure between the assets. The training appears to converge, but the average reward remains around 175 per RFQ, below the roughly 195 reported in the paper.

In the five-bond case, the actor--critic implementation becomes much harder to train reliably. In our experiments, the average reward stabilizes around 300 per RFQ and does not improve as much as hoped.

The report also studies a simpler greedy alternative for policy improvement. This variant remains close to the original method in the one-bond and two-bond settings, and in the five-bond case it reaches a more satisfactory average reward of around 350 per RFQ under the reported experimental setup.

## The greedy variant

The critic is the strongest part of the original methodology because, in this model-based setting, it already gives access to the continuation value of each state. Once the expected one-step reward and next-state values are known, it is natural to define an action-value function and improve the policy directly over a discrete grid of admissible trade probabilities.

This greedy improvement step is much simpler than learning a second neural network for the policy. It is also more interpretable. Its limitation is scalability: once the number of bonds grows, the state-action space becomes too large and the tabular approach loses its appeal. For that reason, it should be viewed as a useful low-dimensional alternative rather than a replacement for the original actor--critic framework.

## Methodologies available

The repository now supports two training methodologies.

### 1. Classic
This is the paper-style actor--critic implementation:
- one symmetric neural actor per bond,
- one critic neural network,
- actor outputs probabilities of trade,
- probabilities are converted into quotes through the execution model,
- actor is improved through the perturbation-based update used in the notebook,
- critic is updated by TD learning from long and short rollouts.

### 2. Greedy
This is the alternative low-dimensional methodology:
- one table actor per bond,
- one critic neural network,
- critic is updated exactly as above,
- the actor is not updated by gradient descent,
- instead, after each critic step, the policy is greedily refreshed over a grid of admissible trade probabilities.

## Pretraining logic

### Single-bond case
The single-bond pipeline follows the notebook:
- actor pretraining on the myopic probability,
- critic pretraining from a finite-difference approximation of the 1D value function,
- final training with long and short rollouts and the risk-limit schedule.

### Multi-bond case
For the classic methodology, multi-bond warm-start relies on stored learned 1D datasets:
- each single-bond run exports learned values and learned bid/ask quote curves,
- multi-bond critic pretraining uses the additive approximation  
  `V(q) ≈ Σ_i V_i(q_i)`,
- each actor is pretrained from the corresponding stored 1D quote curve.

For the greedy methodology, the current implementation keeps the table-actor spirit of the alternative notebook:
- critic pretraining from finite-difference value grids,
- table actors initialized from myopic probabilities,
- greedy policy refresh during final training.

## Output structure

Outputs are separated by methodology so the two approaches do not overwrite one another.

```text
outputs/
├─ classic/
│  ├─ single_bond/
│  │  └─ bond_XX/
│  │     ├─ targets.npz
│  │     ├─ metrics.json
│  │     ├─ model_checkpoint.pt
│  │     └─ plots/
│  └─ multi_bond/
│     └─ bonds_XX_YY.../
│        ├─ metrics.json
│        ├─ model_checkpoint.pt
│        └─ plots/
└─ greedy/
   ├─ single_bond/
   │  └─ bond_XX/
   │     ├─ targets.npz
   │     ├─ metrics.json
   │     ├─ model_checkpoint.pt
   │     └─ plots/
   └─ multi_bond/
      └─ bonds_XX_YY.../
         ├─ metrics.json
         ├─ model_checkpoint.pt
         └─ plots/
         
## How to run

### Classic methodology

#### Single-bond

```bash
python scripts/run_single_bond.py --bond 0 --methodology classic
python scripts/plot_single_bond.py --bond 0 --methodology classic
```

#### Multi-bond

```bash
python scripts/run_multi_bond.py --bonds 0 5 --methodology classic --hidden_dim 12 --nb_steps 500
python scripts/plot_multi_bond_learning.py --bonds 0 5 --methodology classic
python scripts/plot_two_bond_surfaces.py --bonds 0 5 --methodology classic
```

### Greedy methodology

#### Single-bond

```bash
python scripts/run_single_bond.py --bond 0 --methodology greedy
python scripts/plot_single_bond.py --bond 0 --methodology greedy
```

#### Multi-bond

```bash
python scripts/run_multi_bond.py --bonds 0 5 --methodology greedy --hidden_dim 12 --nb_steps 500
python scripts/plot_multi_bond_learning.py --bonds 0 5 --methodology greedy
python scripts/plot_two_bond_surfaces.py --bonds 0 5 --methodology greedy
```
