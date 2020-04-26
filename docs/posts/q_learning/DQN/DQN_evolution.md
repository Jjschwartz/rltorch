# The evolution of Deep Q-Networks overtime

Deep Q-networks (DQN) came into prominence in 2013 when a team from DeepMind showed how Q-learning and Deep learning could be combined to produce human level performance in a range of Atari games [1]. Since then there have been a number of improvements made that have pushed the high scores of DQN agents further and further.

## DQN

The original paper on DQN was the first to successfully combine Q-learning and Deep learning and to learn to play a number of complex games from raw pixels [1]. This key contributions of this paper were:

1. Using a Neural Network (specifically a CNN) as the function approximator for the Q-value
2. Combine this with experience replay
3. Showed how many tricks are required to get Deep RL to work in practice.

Using a neural network greatly improved the generalization capability of the RL agents and allowed them to learn in such a high dimensional environment. Using a Convolutional Neural Network specifically allowed the agents to efficiently learn from images. As a side note, this was not not the first to combine Q-learning and neural networks (that was TD-Gammon), but they were the first to use a deep architecture and apply it successfully in a complex domain.

The use of experience replay, had a number of key benefits. Firstly, it allowed for the reuse of each step of experience in potentially multiple network updates, improving the efficiency if the algorithm. Secondly, it removed the correlation betweek consecutive training samples by random sampling from the replay. Thirdy, it helps to decouple training data generation and the training data used for the next policy update (when compared with on-policy training).

## DQN with a Target Network

[2]

## Double DQN

[3]

## Prioritised experience replay

[4]

## Dueling

[5]

## Distributed





## References

[1] The original DQN paper:

    [https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf](Mnih et al (2013) Playing Atari with Deep Reinforcement Learning)

[2] Extended DQN with a target network

    [https://www.nature.com/articles/nature14236.pdf](Mnih et al (2015) Human-level control through deeo reinforcement learning)


[3] Double DQN

    [https://arxiv.org/abs/1509.06461](Hasselt et al (2015) Deep reinforcement learning with double Q-learning)

[4] Dueling DQN

    [https://arxiv.org/abs/1511.06581](Dueling network architectures for deep reinforcement learning)

[5] Prioritised experience replay

    [https://arxiv.org/abs/1511.05952](Prioritized experience replay)
