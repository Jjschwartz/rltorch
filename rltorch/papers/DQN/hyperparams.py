"""Hyperparameters from paper """
import numpy as np
import torch.optim as optim


class AtariHyperparams:

    ALGO = "DQN"
    SEED = 2

    LOG_DISPLAY_FREQ = 10

    # Image sizing
    WIDTH = 84
    HEIGHT = 84
    # Number of most recent frames given as input to Q-network
    AGENT_HISTORY = 4
    STATE_DIMS = (AGENT_HISTORY, WIDTH, HEIGHT)
    NORMALIZE = False

    DISCOUNT = 0.99
    MINIBATCH_SIZE = 32
    REPLAY_SIZE = int(1e6)
    REPLAY_S_DTYPE = np.uint8
    # Number of network updates between target network updates
    # TARGET_NETWORK_UPDATE_FREQ = 10000
    TARGET_NETWORK_UPDATE_FREQ = 2500
    # Number of times an action is repeated, i.e. number of frames skipped
    ACTION_REPEAT = 4
    # Num actions (ignoring repeats) performed before Gradient descent update
    NETWORK_UPDATE_FREQUENCY = 4

    # Parameters for network learning
    OPTIMIZER = optim.RMSprop
    LEARNING_RATE = 0.00025
    GRADIENT_MOMENTUM = 0.95
    SQUARED_GRADIENT_MOMENTUM = 0.95
    MIN_SQUARED_GRADIENT = 0.01
    OPTIMIZER_KWARGS = {
        "lr": LEARNING_RATE,
        "momentum": GRADIENT_MOMENTUM,
        "eps": MIN_SQUARED_GRADIENT
    }
    GRAD_CLIP = [-1, 1]

    # for reward
    R_CLIP = [-1, 1]

    # Exploration
    EXPLORATION_SCHEDULE = "Linear"
    INITIAL_EXPLORATION = 1.0
    FINAL_EXPLORATION = 0.1
    FINAL_EXPLORATION_FRAME = 1000000
    # Number of frames to run random policy and before learning starts
    REPLAY_START_SIZE = 50000
    # Max number of "do nothing" actions to be performed at start of episode
    NO_OP_MAX = 30

    # Network architecture
    INPUT_DIMS = (WIDTH, HEIGHT, AGENT_HISTORY)
    LAYER_1 = {"type": "convolutional",
               "filters": 32, "kernel_size": (8, 8),
               "stride": 4, "activation": "relu"}
    LAYER_2 = {"type": "convolutional",
               "filters": 64, "kernel_size": (4, 4),
               "stride": 2, "activation": "relu"}
    LAYER_3 = {"type": "convolutional",
               "filters": 64, "kernel_size": (3, 3),
               "stride": 1, "activation": "relu"}
    LAYER_4 = {"type": "fully_connected",
               "size": 512, "activation": "relu"}
    OUTPUT = {"type": "fully_connected"}

    # training duration (50 million)
    TRAINING_FRAMES = int(5e7)

    # Other hyperparams not related to paper
    # Model Save Freq
    MODEL_SAVE_FREQ = int(1e6)

    # Evaluation
    EVAL_FREQ = int(1e6)
    EVAL_STEPS = 125000
    EVAL_EPSILON = 0.05

    @classmethod
    def set_seed(cls, seed):
        cls.SEED = seed

    @classmethod
    def set_mode(cls, mode='dqn'):
        if mode == "testing":
            print("WARNING: using test hyperparams")
            input("Press any key to continue..")
            cls.ALGO += "_test"
            cls.REPLAY_SIZE = int(1e4)
            cls.REPLAY_START_SIZE = 100
            cls.INITIAL_EXPLORATION = 0.1
            cls.TARGET_NETWORK_UPDATE_FREQ = 1000
            cls.EVAL_FREQ = 2000
            cls.EVAL_STEPS = 1000
            cls.MODEL_SAVE_FREQ = 2500
            cls.LOG_DISPLAY_FREQ = 1
            cls.MINIBATCH_SIZE = 12
        elif mode == "eval":
            cls.ALGO += "_eval"
            cls.REPLAY_SIZE = int(1e4)
        elif mode == "ddqn":
            print("Using DDQN hyperparams")
            cls.ALGO = "DDQN"
        elif mode == "ddqn-tuned":
            print("Using DDQN-Tuned hyperparams")
            cls.ALGO = "DDQN-Tuned"
            cls.TARGET_NETWORK_UPDATE_FREQ = 30000
            cls.FINAL_EXPLORATION = 0.01
            cls.EVAL_EPSILON = 0.001
        elif mode == "dqn":
            print("Using DQN hyperparams")
            pass
        elif mode == "normalized":
            print("Using normalized observations")
            cls.NORMALIZE = True
            cls.REPLAY_S_DTYPE = np.float16
        elif mode == "pong_tuned":
            print("Using pong tuned hyperparams")
            cls.REPLAY_SIZE = 100000
            cls.REPLAY_START_SIZE = 10000
            cls.INITIAL_EXPLORATION = 1.0
            cls.FINAL_EXPLORATION = 0.02
            cls.FINAL_EXPLORATION_FRAME = 100000
            # this corresponds to updating every 1000 frames
            cls.TARGET_NETWORK_UPDATE_FREQ = 250
            cls.OPTIMIZER = optim.Adam
            cls.OPTIMIZER_KWARGS = {"lr": 1e-4}
        else:
            raise ValueError("Unsupported Hyper param mode")

    @classmethod
    def get_all_hyperparams(cls):
        all_kwargs = {}
        for k, v in cls.__dict__.items():
            if not any([k.startswith("__"),
                        isinstance(v, classmethod)]):
                all_kwargs[k] = v
        return all_kwargs
