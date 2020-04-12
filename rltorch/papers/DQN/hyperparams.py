"""Hyperparameters from paper """

TESTING = True

# Image sizing
WIDTH = 84
HEIGHT = 84
# Number of most recent frames given as input to Q-network
AGENT_HISTORY = 4
STATE_DIMS = (AGENT_HISTORY, WIDTH, HEIGHT)

DISCOUNT = 0.99
MINIBATCH_SIZE = 32
REPLAY_SIZE = int(1e6)
# Number of steps between target network updates
TARGET_NETWORK_UPDATE_FREQ = 10000
# Number of times an action is repeated, i.e. number of frames skipped
ACTION_REPEAT = 4
# How many actions are performed before Gradient descent update
NETWORK_UPDATE_FREQUENCY = 4

# Parameters for network learning
OPTIMIZER = "RMS Prop"
LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
SQUARED_GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01
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
EVAL_FREQ = 250000
EVAL_STEPS = 125000
EVAL_EPSILON = 0.05

if TESTING:
    print("WARNING: using test hyperparams")
    input("Press any key to continue..")
    REPLAY_SIZE = int(1e5)
    REPLAY_START_SIZE = 1000
    EVAL_FREQ = 1000
    EVAL_STEPS = 1000


ALL_KWARGS = locals()
for k in list(ALL_KWARGS.keys()):
    if k.startswith("__"):
        ALL_KWARGS.pop(k)
