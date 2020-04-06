from rltorch.tuner.random_tuner import RandomTuner
from rltorch.algs.q_learning.DQN.agent import DQNAgent


hyperparams = {
    # constants
    "training_steps": [100000],
    "final_epsilon": [0.01],
    "init_epsilon": [1.0],
    "exploration": [1000],
    "gamma": [0.999],
    "start_steps": [32],
    "network_update_freq": [1],
    "model_save_freq": [None],
    # to sample
    "hidden_sizes": [[64], [64, 64], [64, 64, 64]],
    "lr": [0.01, 0.001, 0.0001],
    "batch_size": [1, 32, 128],
    "replay_size": [1000, 10000, 100000],
    "target_update_freq": [100, 1000, 10000],
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='CartPole-v0')
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--num_cpus", type=int, default=1)
    args = parser.parse_args()

    tuner = RandomTuner(name=args.exp_name, num_exps=args.num_runs)

    hyperparams["env_name"] = [args.env_name]
    for k, v in hyperparams.items():
        tuner.add(k, v)

    tuner.run(DQNAgent, args.num_cpus)
