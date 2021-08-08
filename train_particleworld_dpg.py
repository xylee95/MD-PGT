import argparse
import gym
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pfrl
import pdb
import model
import envs
from envs import make_particleworld
import update_functions
from update_functions import *

parser = argparse.ArgumentParser(description='PyTorch example')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--dim', type=int, default=5, help='Number of dimension')
parser.add_argument('--num_agents', type=int, default=5, help='Number of agents')
parser.add_argument('--max_eps_len', type=int, default=50, help='Number of steps per episode')
parser.add_argument('--num_episodes', type=int, default=50000, help='Number training episodes')
parser.add_argument('--env', type=str, default='particle_world', help='Training env')
parser.add_argument('--gpu', type=bool, default=False, help='Enable GPU')
parser.add_argument('--opt', type=str, default='sgd', help='Optimizer')
parser.add_argument('--momentum', type=float, default=0.0, help='Momentum term for SGD')
parser.add_argument('--topology', type=str, default='dense', choices=('dense','ring','bipartite'))

args = parser.parse_args()
torch.manual_seed(args.seed)

def main():
    num_agents = args.num_agents
    dimension = args.dim
    fpath = os.path.join('dpg_results', args.env, str(dimension) + 'D', args.opt + str(args.topology))
    if not os.path.isdir(fpath):
        os.makedirs(fpath)

    # initliaze multiple agents and optimizer
    if args.gpu:
        device = 'cuda:0'
    else:
        device = 'cpu'

    sample_env = make_particleworld.make_env('simple_spread', num_agents=num_agents, num_landmarks=dimension)
    sample_env.discrete_action_input = True #set action space to take in discrete numbers 0,1,2,3
    print('observation Space:', sample_env.observation_space)
    print('Action Space:', sample_env.action_space)
    print('Number of agents:', sample_env.n)
    sample_obs = sample_env.reset()
    sample_obs = np.concatenate(sample_obs).ravel().tolist()

    agents = []
    optimizers = []
    for i in range(num_agents):
        agents.append(model.Policy(state_dim=len(sample_obs), action_dim=4).to(device))
        optimizers.append(optim.SGD(agents[i].parameters(), lr=3e-4, momentum=0.0))

    pi = load_pi(num_agents=args.num_agents, topology=args.topology)
    # RL setup
    num_episodes = args.num_episodes
    max_eps_len = args.max_eps_len
    done = False
    R = 0 
    R_hist = []
    R_hist_plot = []

    env = make_particleworld.make_env('simple_spread', num_agents=num_agents, num_landmarks=dimension)
    env.discrete_action_input = True #set action space to take in discrete numbers 0,1,2,3

    for episode in range(num_episodes):
        obs = env.reset()
        obs = np.concatenate(obs).ravel().tolist()
        for t in range(1, max_eps_len):
            actions = []
            for policy in agents:
                action = model.select_action(obs, policy)
                actions.append(action)
            #print('step:', t)
            obs, rewards, done_n, _ = env.step(actions)
            obs = np.concatenate(obs).ravel().tolist()
            done = all(item == True for item in done_n)

            for i in range(len(agents)):
                #print('r:', rewards[i])
                agents[i].rewards.append(rewards[i])
            
            R += sum(rewards)
            reset = t == max_eps_len-1
            if done or reset:
                #print(f'Eps: {episode} Done: {done_n} Reset:{reset} State:{obs} reward:{rewards}')
                print(f'Eps: {episode} Done: {done_n} Reset:{reset} reward:{rewards}')
                R_hist.append(R)
                R = 0
                break

        for policy, optimizer in zip(agents, optimizers):
            _ = compute_grads(args, policy, optimizer)

        #agents = global_average(agents, num_agents)
        agents = take_param_consensus(agents, pi)

        for policy, optimizer in zip(agents, optimizers):
            update_weights(policy, optimizer)

        if episode % args.log_interval == 0:
            avg_reward = np.sum(R_hist)/len(R_hist)
            R_hist_plot.append(avg_reward)
            R_hist = []
            print(f'Episode:{episode} Average reward:{avg_reward:.2f}')
                            
    plt.figure()
    plt.plot(R_hist_plot)
    plt.ylabel('Reward')
    plt.xlabel('Episodes')
    plt.title(str(dimension) + '-d ' + ' ' + args.env + args.opt + ' ' + str(args.seed))
    plt.savefig(os.path.join(fpath, str(args.seed) + '_R.jpg'))
    plt.close()

    np.save(os.path.join(os.path.join(fpath, 'R_array_' + str(args.seed) + '.npy')), R_hist_plot)

if __name__ == '__main__':
    main()