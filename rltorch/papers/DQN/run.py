
from rltorch.papers.DQN.dqn_agent import DQNAgent

env_name = "pong"

agent = DQNAgent(env_name)
agent.train()
