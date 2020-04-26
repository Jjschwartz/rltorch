from .ddqn_agent import DDQNAgent


class DuelingDQNAgent(DDQNAgent):
    """Nothing to do apart from inherit from DDQNAgent, the rest is handled
    within the hyperparam class, where the DuelingDQN model is set.

    This class is only really used for clarity.
    """
    pass
