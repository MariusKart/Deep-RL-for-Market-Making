# Deep Reinforcement Learning for Market Making in Corporate Bonds

This repository contains a reproduction of the paper *Deep Reinforcement Learning for Market Making in Corporate Bonds* by Guéant and Manziuk, together with a simple policy-improvement variant developed as part of the Reinforcement Learning coursework in the MSc Mathematics and Finance at Imperial College London.

The project studies market making on multi-dealer-to-client corporate bond platforms. A dealer receives RFQs, chooses bid and ask quotes, earns spread when trades occur, and at the same time must control inventory risk. In low dimension, this problem can be approached with classical stochastic control and HJB methods. In higher dimension, these methods become difficult to use in practice because the inventory state space grows too quickly.

The main idea of the paper is to replace direct PDE-based resolution with a model-based reinforcement learning procedure. The market-making problem is written in discrete time at RFQ arrival times, and the quoting rule is learned with an actor--critic architecture. A key point is that the actor does not output quotes directly: it outputs probabilities of trade, which are then converted back into spreads through the execution model.

## What is implemented here

The repository reproduces the main reinforcement learning framework of the paper:
- a simulated market-making environment for correlated corporate bonds,
- inventory-based state dynamics,
- a critic network used to approximate the value function,
- an actor network used to parameterize the quoting policy through trade probabilities,
- training through rollouts, TD learning, and policy-improvement updates.

The code also includes an additional policy-improvement approach proposed in the coursework report. Instead of learning the actor with random perturbations, the critic is kept fixed and the policy is improved by direct maximization of the one-step action-value over a discretized grid of admissible trade probabilities. This produces a simple greedy alternative that is easier to interpret in low dimension.

## Financial intuition

At the core of the problem is the usual market-making trade-off. Tight quotes increase the probability of trading and therefore the expected spread capture, but they also make inventory more volatile. Wide quotes reduce trading intensity and help control exposure. The optimal policy must therefore adapt to the current inventory and, in the multi-bond case, to the interaction between correlated positions.

In the one-bond case, the learned policy has the expected shape: when inventory is too short, bid quotes become more aggressive in order to buy back inventory; when inventory is already too long, bid quotes become less aggressive in order to avoid accumulating more risk. In the two-bond case, the learned surfaces reflect the fact that inventory in one bond can partly hedge exposure in another correlated bond.

## Results

The main reproduction results are the following.

In the one-bond case, the reinforcement learning procedure recovers the expected policy shape and reaches an average reward of about 200 per RFQ, which is consistent with the benchmark discussed in the report.

In the two-bond case, using bonds 1 and 6, the learned value function and quote surfaces reflect the correlation structure between the assets. The training appears to converge, but the average reward remains around 175 per RFQ, below the roughly 195 reported in the paper. In the report, we discuss several possible explanations for this gap, including fewer training steps and uncertainty around the pre-training procedure.

In the five-bond case, the actor--critic implementation becomes much harder to train reliably. In our experiments, the average reward stabilizes around 300 per RFQ and does not improve as much as hoped.

The report also studies a simpler greedy alternative for policy improvement. This variant remains close to the original method in the one-bond and two-bond settings, and in the five-bond case it reaches a more satisfactory average reward of around 350 per RFQ under the reported experimental setup.

## The greedy variant

The critic is the strongest part of the original methodology because, in this model-based setting, it already gives access to the continuation value of each state. Once the expected one-step reward and next-state values are known, it is natural to define an action-value function and improve the policy directly over a discrete grid of admissible trade probabilities.

This greedy improvement step is much simpler than learning a second neural network for the policy. It is also more interpretable. Its limitation is scalability: once the number of bonds grows, the state-action space becomes too large and the tabular approach loses its appeal. For that reason, it should be viewed as a useful low-dimensional alternative rather than a replacement for the original actor--critic framework.

## Repository structure

```text
.
├── core/          # simulation environment and market-making mechanics
├── notebooks/     # numerical experiments and visualizations
├── training/      # training routines for critic / actor
├── config/        # constants and model parameters
└── README.md
