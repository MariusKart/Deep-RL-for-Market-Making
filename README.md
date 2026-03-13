# Deep RL for Market Making

This repository contains an initial implementation of a coursework project for the **Reinforcement Learning** module (Spring 2026) in the **MSc Mathematics and Finance** at **Imperial College London**.

The project aims to reproduce and explore the approach in **Guéant et al. (2019)**, applying **deep reinforcement learning** to the problem of **market making**. The focus is on designing an agent that learns bid/ask quoting strategies while managing inventory risk in a simulated trading environment.

## Project Status

This project is currently a **work in progress**.  
The repository contains the initial setup, core ideas, and ongoing implementation. Further development will include model refinement, training, evaluation, and comparison with the reference paper.

## Objectives

- Reproduce the main ideas of Guéant et al. (2019)
- Formulate market making as a reinforcement learning problem
- Implement a simulated trading environment
- Train a deep RL agent to learn quoting strategies
- Evaluate the agent in terms of profitability and inventory management

## Problem Overview

Market makers provide liquidity by posting buy and sell quotes. Their objective is typically to earn the bid-ask spread while controlling exposure to inventory risk. This makes market making a natural sequential decision-making problem, where reinforcement learning can be used to learn adaptive policies under uncertainty.

## Current Repository Structure

```text
.
├── core/          # contains the simulation environment and functions
├── notebooks/     # numerical experimentations
├── training/      # training functions      
├── config/        # constants and training parameters
├── README.md