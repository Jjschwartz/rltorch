# Reinforcement Learning

In this post I want to explore what is Reinforcement Learning (or RL for short), what is the goal of RL, why care about it, what are the different aspects of RL and how do all the different types (e.g. off-policy vs on-policy, model-freevs model-based) of RL relate?

## What is the problem RL is trying to solve?

Imagine we have a complex system or environment and in this system we are trying to achieve some goal



## What are the components of RL?

At the highest level any RL problem requires these three components:

1. An agent (i.e. the decision maker)
2. An Environment, which can be interacted with (i.e. we can perform actions in) and we can observe the result of this interaction
3. A reward function that provides some real value for each given action and observation and defines the objective of the agent

I seperated the environment and the reward function, since for the same environment we could want to achieve different tasks. The reward function is something that is defined for a given problem.

The RL agent is itself made up of a number of components. Which components are included and the shape they take is what seperates different types of RL Agents (and their underlying algorithms).

One thing all RL agents must have is a **policy**, which is simply a function which determines which action to perform for any given observation. A policy can be extremely simple (e.g. always perform the same action for any observation) or complex (e.g. a neural network) and also deterministic (always returns same action for a given observation) or stochastic (returns an action sampled from a distribution for a given observation). Additionally, the policy could remain constant or be learnt overtime. Obviously, a policy that doesn't lead to different behaviour overtime will be very limited in its ability to improve.

Another component that is ubiquitous in RL is the **value function**. The value function is a function that returns the expected total reward tha agent will recieve starting from a given state given the agents current policy. It essentially describes how good a state is in the long term. Similar to the policy it can have a simple form (e.g. a look up table) or complex (e.g. a neural net). Technically, it is possible to have a RL agent without a value function depending on the specific approach (e.g. the REINFORCE policy gradient algorithm) but in practice they are generally always helpful.

The last key component that may be present in a RL agent is a **model** of the environment. A model is again a function that maps a given state and action to the next state (and reward). whether an RL agent uses a model is a key distinguishing feature of RL algorithms (see Model-based vs Model-free section below).


## Model-based vs Model-free RL
