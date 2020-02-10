"""Simple test of image preprocessing """
import gym
import numpy as np
from rltorch.papers.DQN.preprocess import ImageProcessor


processor = ImageProcessor(84, 84)

env = gym.make("Pong-v0")
num_actions = env.action_space.n

o1 = env.reset()
d = False
while not d:
    a = np.random.choice(num_actions)
    o2, _, d, _ = env.step(a)
    processor.debug(o1, o2)
    o1 = o2
    input("Hit enter for next step")
