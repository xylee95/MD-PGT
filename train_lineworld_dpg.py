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
from envs import lineworld
import update_functions
from update_functions import *

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99,
					help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='interval between training status logs (default: 10)')
parser.add_argument('--dim', type=int, default=2, help='Number of dimension')
parser.add_argument('--num_agents', type=int, default=2, help='Number of agents')
parser.add_argument('--max_eps_len', type=int, default=500, help='Number of steps per episode')
parser.add_argument('--num_episodes', type=int, default=5000, help='Number training episodes')
parser.add_argument('--env', type=str, default='lineworld', help='Training env')
parser.add_argument('--gpu', type=bool, default=False, help='Enable GPU')
parser.add_argument('--opt', type=str, default='sgd', help='Optimizer')
parser.add_argument('--momentum', type=float, default=0.0, help='Momentum term for SGD')
parser.add_argument('--topology', type=str, default='dense', choices=('dense','ring','bipartite'))

args = parser.parse_args()
torch.manual_seed(args.seed)

def main():
	# initialize env
	num_agents = args.num_agents
	dimension = args.dim
	fpath = os.path.join('dpg_results', args.env, str(dimension) + 'D', args.opt)
	if not os.path.isdir(fpath):
		os.makedirs(fpath)

	env = lineworld.LineWorld(dimension=dimension, seed=args.seed)

	# initliaze multiple agents and optimizer
	if args.gpu:
		device = 'cuda:0'
	else:
		device = 'cpu'

	agents = []
	optimizers = []
	for i in range(num_agents):
		agents.append(model.Policy(state_dim=dimension, action_dim=3).to(device))
		optimizers.append(optim.SGD(agents[i].parameters(), lr=3e-4, momentum=args.momentum))

	pi = load_pi(num_agents=args.num_agents, topology=args.topology)
	# RL setup
	num_episodes = args.num_episodes
	done = False
	max_eps_len = args.max_eps_len
	R = 0 
	R_hist = []
	R_hist_plot = []

	for episode in range(num_episodes):
		state = env.reset()
		state = torch.FloatTensor(state).to(device)
		for t in range(1, max_eps_len):  # Don't infinite loop while learning
			actions = []
			for policy in agents:
				action = model.select_action(state, policy)
				actions.append(action)
			actions = torch.as_tensor([actions])
	
			#step through enviroment with set of actions. rewards is list of reward
			state, rewards, done = env.step(actions)
			for i in range(len(agents)):
				#print('r:', rewards[i])
				agents[i].rewards.append(rewards[i])
				
			state = torch.FloatTensor(state).to(device)
			R += np.sum(rewards)
			reset = t == max_eps_len-1
			if done or reset:
				print(f'Done: {done} Reset:{reset} State:{state} reward:{rewards}')
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