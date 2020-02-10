"""Hyperparameters from paper """

# Image sizing
WIDTH = 84
HEIGHT = 84
FRAMES = 4
STATE_DIMS = (FRAMES, WIDTH, HEIGHT)

MINIBATCH_SIZE = 32
REPLAY_SIZE = 1000000
# Number of most recent frames given as input to Q-network
AGENT_HISTORY = 4
TARGET_NETWORK_UPDATE_FREQ = 10000
DISCOUNT = 0.99
# Number of times an action is repeated, i.e. number of frames skipped
ACTION_REPEAT = 4
# Network update frequency. How many actions are performed before Gradient descent update
NETWORK_UPDATE_FREQUENCY = 4

# Parameters for network learning
OPTIMIZER = "RMS Prop"
LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
SQUARED_GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01

# Exploration
EXPLORATION_SCHEDULE = "Linear"
INITIAL_EXPLORATION = 1.0
FINAL_EXPLORATION = 0.1
FINAL_EXPLORATION_FRAME = 1000000
# Number of frames to run random policy and before learning starts
REPLAY_START_SIZE = 50000
# Max number of "do nothing" actions to be performed by agent at start of episode
NO_OP_MAX = 30

# Network architecture
INPUT_DIMS = (WIDTH, HEIGHT, AGENT_HISTORY)
LAYER_1 = {"type": "convolutional", "filters": 32, "kernel_size": (8, 8), "stride": 4, "activation": "relu"}
LAYER_2 = {"type": "convolutional", "filters": 64, "kernel_size": (4, 4), "stride": 2, "activation": "relu"}
LAYER_3 = {"type": "convolutional", "filters": 64, "kernel_size": (3, 3), "stride": 1, "activation": "relu"}
LAYER_4 = {"type": "fully_connected", "size": 512, "activation": "relu"}
OUTPUT = {"type": "fully_connected"}

# for reward
R_CLIP = [-1, 1]
