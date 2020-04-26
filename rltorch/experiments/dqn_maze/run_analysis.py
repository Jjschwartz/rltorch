import gym
import torch
import random
import gym_maze
import numpy as np

from rltorch.experiments.dqn_maze.agent import DQNMazeAgent


def get_random_action(num_actions):
    return random.randint(0, num_actions-1)


def one_hot_encode(o, maze_size):
    o = o.reshape(-1)
    one_hot = np.zeros(maze_size*2)
    for d, i in enumerate(o):
        one_hot[d*maze_size + int(i)] = 1.0
    return one_hot


def collect_states(env, steps=10000, policy=None, use_one_hot=False):
    """Collect states from env """
    print("Collecting states")
    states = dict()
    t = 0
    s = ""
    done = True
    maze_size = env.observation_space.high[0]+1
    print(maze_size)

    def random_policy(s):
        return get_random_action(env.action_space.n)

    if policy is None:
        policy = random_policy

    while t < steps:
        if done:
            s = env.reset()
        s_key = str(s)
        if use_one_hot:
            s = one_hot_encode(s, maze_size)
        states[s_key] = s
        a = policy(s)
        s, _, done, _ = env.step(a)
        t += 1
        if t > 0 and t % int(steps//10) == 0:
            print(f"Step = {t}")

    print(f"Finished. {len(states)} states collected.")
    return states


def visualize_qvals(env, qfunc, states):
    """Visualize the q function for given states """
    print("\nVisualizing Q-Values")
    for s_key in sorted(states.keys()):
        q_vals = qfunc(states[s_key])
        print(f"{s_key} = {q_vals}. Best a = {q_vals.argmax()}")
    print()


def get_dqn_qfunc(agent):
    def qfunc(s):
        with torch.no_grad():
            s = torch.from_numpy(s).float().to(agent.device)
            return agent.dqn(s).numpy()
    return qfunc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default='CartPole-v0')
    parser.add_argument("--hidden_sizes", type=int, nargs="*",
                        default=[64, 64])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--training_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay_size", type=int, default=100000)
    parser.add_argument("--final_epsilon", type=float, default=0.05)
    parser.add_argument("--init_epsilon", type=float, default=1.0)
    parser.add_argument("--exploration", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--start_steps", type=int, default=32)
    parser.add_argument("--network_update_freq", type=int, default=1)
    parser.add_argument("--target_update_freq", type=int, default=1000)
    parser.add_argument("--model_save_freq", type=int, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--collect_steps", type=int, default=10000)
    parser.add_argument("--viz_freq", type=int, default=10000)
    parser.add_argument("--use_one_hot", action="store_true")
    parser.add_argument("--use_target_network", action="store_true")
    args = parser.parse_args()

    env = gym.make(args.env_name)
    states = collect_states(env,
                            args.collect_steps,
                            use_one_hot=args.use_one_hot)

    print(f"\n{'='*60}\nDQN Analysis\n{'='*60}")
    viz_rounds = int(args.training_steps // args.viz_freq)
    agent = DQNMazeAgent(**vars(args))
    qfunc = get_dqn_qfunc(agent)
    visualize_qvals(states, qfunc, states)
    for i in range(viz_rounds):
        agent.training_steps = (i+1) * args.viz_freq
        agent.train()
        visualize_qvals(states, qfunc, states)
        input("Press key to continue..")
