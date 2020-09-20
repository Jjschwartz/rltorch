# rltorch
Deep RL algorithm implementations using pytorch

## Installation

Clone the repo:

Using `HTTPS`:

```
$ git clone https://github.com/Jjschwartz/rltorch.git
```

Or using `SSH`:

```
$ git clone git@github.com:Jjschwartz/rltorch.git
```

Install:

```
$ cd rltorch
$ pip install -e .
```

## Demo

To run the DDQN algorithm for the *CartPole-v0* environment, from the base `rltorch` directory run:

```
$ python rltorch/algs/q_learning/DDQN/run.py --env_name CartPole-v0 --render_last
```

This will train a DDQN agent, which should take 2-5 minutes depending on your PC, and then display the final policy in action.

To view the available hyperparameters you can run the program with the `--help` flag:

```
$ python rltorch/algs/q_learning/DDQN/run.py --help
```

This will display all options and their default values.

The other algorithms can be run in the same way, just change the directory of the run script:

1. DQN - `algs/q_learning/DQN/run.py`
2. DQN with target Network - `algs/q_learning/DQNTarget/run.py`
3. DDQN - `algs/q_learning/DDQN/run.py`
4. Dueling DQN - `algs/q_learning/DDQN/run.py`

**Note:** There are additional algorithms implemented but their interfaces are still a work in progress.

## Algorithms

### Q-learning
1. DQN
2. DQN with target Network
3. Double DQN
4. Dueling DQN

### Policy Gradients
1. REINFORCE
2. PPO (and PPO using LSTM)

## Papers

Papers reproduced.

1. DQN - [V Mnih et al (2013) Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
2. DQN with target Network - [V Mnih et al (2015) Human-level control through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236.pdf)
3. Double DQN - [H van Hasselt et al (2015) Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
4. Dueling DQN - [Z Wang et al (2015) Dueling network architectures for deep reinforcement learning](https://arxiv.org/abs/1511.06581)

## References

1. Spinningup by OpenAI: [spinningup.openai.com](https://spinningup.openai.com)
