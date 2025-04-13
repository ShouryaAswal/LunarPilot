# LunarPilot

LunarPilot is a reinforcement learning project that trains agents to master the OpenAI Gymnasium's LunarLander-v3 environment. This repository implements two different evolutionary optimization techniques to train intelligent landing agents:

1. **Genetic Algorithm (GA)** with Simulated Binary Crossover and Polynomial Mutation
2. **Proximal Swarm Optimization (PSO)** with Lévy Flight exploration

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Training Methods](#training-methods)
  - [Genetic Algorithm](#genetic-algorithm)
  - [Proximal Swarm Optimization](#proximal-swarm-optimization)
- [Usage](#usage)
  - [Training an Agent](#training-an-agent)
  - [Playing with a Trained Agent](#playing-with-a-trained-agent)
  - [Evaluating Agent Performance](#evaluating-agent-performance)
- [How It Works](#how-it-works)
- [License](#license)

## Overview

LunarPilot teaches an agent to land a spacecraft on the moon using a neural network policy optimized by evolutionary algorithms. The goal is to safely navigate and land on the landing pad without crashing or using excessive fuel.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/LunarPilot.git
   cd LunarPilot
   ```

2. Install the required dependencies:
   ```bash
   pip install gymnasium numpy
   ```

## Training Methods

### Genetic Algorithm

The Genetic Algorithm implementation (`Genetic_Algorithm.py`) features:

- Neural network policy representation (8×4 weight matrix + 4 biases)
- Population-based optimization (configurable population size)
- Elite selection for preserving top performers
- Simulated Binary Crossover (SBX) for genetic recombination
- Polynomial Mutation for exploring the parameter space
- Progressive improvement over multiple generations

### Proximal Swarm Optimization

The PSO implementation (`Proximal_Swarm_Optimization.py`) includes:

- Swarm-based parameter optimization
- Parallel policy evaluation for faster training
- Lévy flight exploration to escape local optima
- Cognitive and social acceleration parameters for balanced exploration/exploitation
- Velocity clamping to prevent unstable behavior
- Continuous improvement tracking with progress visualization

## Usage

### Training an Agent

You can train your agent using either the Genetic Algorithm or PSO:

#### Using Genetic Algorithm:
```bash
python Genetic_Algorithm.py --train --filename ga_best_policy.npy
```

#### Using PSO:
```bash
python Proximal_Swarm_Optimization.py --train --filename pso_best_policy.npy
```

Both commands will generate a `.npy` file containing the trained policy parameters.

### Playing with a Trained Agent

To watch your trained agent in action:

#### For GA-trained agent:
```bash
python Genetic_Algorithm.py --play --filename ga_best_policy.npy
```

#### For PSO-trained agent:
```bash
python Proximal_Swarm_Optimization.py --play --filename pso_best_policy.npy
```

### Evaluating Agent Performance

You can evaluate your agent's performance over multiple episodes using the evaluation script:

```bash
python evaluate_agent.py --filename your_policy.npy --policy_module my_policy
```

The evaluation script will:
1. Load the policy parameters from the specified `.npy` file
2. Import the policy action function from the specified module
3. Run 100 episodes, rendering the first 5 in the human-viewable window
4. Report the average reward across all episodes

## How It Works

Both algorithms optimize a simple neural network policy with:
- 8 input neurons (state observations from LunarLander-v3)
- 4 output neurons (representing possible actions: do nothing, fire left engine, fire main engine, fire right engine)

The policy maps states to actions through:
```
action = argmax(observation · W + b)
```

Where:
- W is an 8×4 weight matrix
- b is a bias vector of length 4

During training, the algorithms evolve these parameters to maximize cumulative reward, creating an effective landing policy.
