import argparse
import gym
import os
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import matplotlib.pyplot as plt

import pfrl
import pdb
import envs
from envs import lineworld
import copy

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99,
					help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='interval between training status logs (default: 10)')
parser.add_argument('--dim', type=int, default=1, help='Number of dimension')
parser.add_argument('--num_agents', type=int, default=2, help='Number of agents')
parser.add_argument('--max_eps_len', type=int, default=500, help='Number of steps per episode')
parser.add_argument('--num_episodes', type=int, default=5000, help='Number training episodes')
parser.add_argument('--env', type=str, default='lineworld', help='Training env')
parser.add_argument('--gpu', type=bool, default=False, help='Enable GPU')
parser.add_argument('--opt', type=str, default='sgd_m', help='Optimizer',
					choices=('adam', 'sgd', 'rmsprop','sgd_m'))
parser.add_argument('--momentum', type=float, default=0.0, help='Momentum term for SGD')
parser.add_argument('--beta', type=float, default=0.9, help='Beta term for surrogate gradient')
parser.add_argument('--min_isw', type=float, default=0.0, help='Minimum value to set ISW')
args = parser.parse_args()
torch.manual_seed(args.seed)

class Policy(nn.Module):
	def __init__(self, state_dim):
		super(Policy, self).__init__()
		self.dense1 = nn.Linear(state_dim, 64)
		self.dense2 = nn.Linear(64, 64)
		self.dense3 = nn.Linear(64, 3)
		self.saved_log_probs = []
		self.rewards = []

	def forward(self, x):
		x1 = torch.tanh(self.dense1(x))
		x2 = torch.tanh(self.dense2(x1))
		x3 = self.dense3(x2)
		dist = Categorical(logits=x3)
		return dist


class SGD_M(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_M, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_M, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None, grads=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        grad_iter = 0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                if grads is not None:
                	d_p = (grads[grad_iter]).view(p.shape)
                	grad_iter+=1
                else:
                	d_p = p.grad
                p.add_(d_p, alpha=-group['lr'])
        return loss
	
def select_action(state, policy):
	try:
		state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
	except:
		pass
	dist = policy(state) # --> Gaussian ( mu | sigma )
	action = dist.sample() # action ~ Gaussian(mu | sigma)
	policy.saved_log_probs.append(dist.log_prob(action)) ## log prob( a | s)
	return action

def compute_grad_traj_prev_weights(state_list, action_list, policy, old_policy, optimizer):

	old_policy_log_probs = []

	for i in range(len(action_list)):
		dist = old_policy(torch.FloatTensor(state_list[i]))
		log_prob = dist.log_prob(action_list[i])
		old_policy_log_probs.append(log_prob)

	eps = np.finfo(np.float32).eps.item()
	R = 0
	policy_loss = []
	returns = []
	for r in policy.rewards[::-1]:
		R = r + args.gamma * R
		returns.insert(0, R)
	returns = torch.tensor(returns)
	returns = (returns - returns.mean()) / (returns.std() + eps)
	for old_log_prob, R in zip(old_policy_log_probs, returns):
		policy_loss.append(-old_log_prob * R)
	
	optimizer.zero_grad() 

	policy_loss = torch.stack(policy_loss).sum() 
	policy_loss.backward()
	# list of tensors gradients, each tensor has shape
	grad = [p.grad.detach().clone().flatten() if (p.requires_grad is True and p.grad is not None)
			else None for group in optimizer.param_groups for p in group['params']]
	return grad

def global_average(agents, num_agents):
	layer_1_w = []
	layer_1_b = []

	layer_2_w = []
	layer_2_b = []

	layer_3_w = []
	layer_3_b = []

	for agent in agents:
		layer_1_w.append(agent.dense1.weight.data)
		layer_1_b.append(agent.dense1.bias.data)

		layer_2_w.append(agent.dense2.weight.data)
		layer_2_b.append(agent.dense2.bias.data)

		layer_3_w.append(agent.dense3.weight.data)
		layer_3_b.append(agent.dense3.bias.data)

	layer_1_w = torch.sum(torch.stack(layer_1_w),0) / num_agents
	layer_1_b = torch.sum(torch.stack(layer_1_b),0) / num_agents

	layer_2_w = torch.sum(torch.stack(layer_2_w),0) / num_agents
	layer_2_b = torch.sum(torch.stack(layer_2_b),0) / num_agents

	layer_3_w = torch.sum(torch.stack(layer_3_w),0) / num_agents
	layer_3_b = torch.sum(torch.stack(layer_3_b),0) / num_agents

	for agent in agents:
		agent.dense1.weight.data = layer_1_w
		agent.dense1.bias.data = layer_1_b

		agent.dense2.weight.data = layer_2_w
		agent.dense2.bias.data = layer_2_b

		agent.dense3.weight.data = layer_3_w
		agent.dense3.bias.data = layer_3_b

	return agents

def compute_grads(policy, optimizer):
	eps = np.finfo(np.float32).eps.item()
	R = 0
	policy_loss = []
	returns = []
	for r in policy.rewards[::-1]:
		R = r + args.gamma * R
		returns.insert(0, R)

	returns = torch.tensor(returns)
	returns = (returns - returns.mean()) / (returns.std() + eps)
	for log_prob, R in zip(policy.saved_log_probs, returns):
		policy_loss.append(-log_prob * R)
	optimizer.zero_grad() 
	policy_loss = torch.stack(policy_loss).sum() 
	print('Initial Loss:',policy_loss)
	policy_loss.backward()
	# list of tensors gradients, each tensor has shape
	grad = [p.grad.detach().clone().flatten() if (p.requires_grad is True and p.grad is not None)
			else None for group in optimizer.param_groups for p in group['params']]
	return grad

def update_weights(policy, optimizer, grads=None):
	if grads is not None:
		optimizer.step(grads=grads)
	else:
		optimizer.step()
	del policy.rewards[:]
	del policy.saved_log_probs[:]

def compute_IS_weight(action_list, state_list, cur_policy, old_policy, min_isw):
	num_list = []
	denom_list = []
	weight_list = []
	# for each policy
	for i in range(len(old_policy)):
		prob_curr_traj = [] # list of probability of every action taken under curreny policy
		# for each step taken
		for j in range(len(action_list)):
			# use save log probability attached to agent
			log_prob = cur_policy[i].saved_log_probs[j][0]
			prob = np.exp(log_prob.detach().numpy())
			prob_curr_traj.append(prob)

		# multiply along all
		prob_tau = np.prod(prob_curr_traj)

		prob_old_traj = []
		# for each step taken
		for j in range(len(action_list)):
			# obtain distribution for given state
			old_dist = old_policy[i](torch.FloatTensor(state_list[j]))
			# compute log prob of action
			# action_list is in the shape of (episode_len, 1, num_agents)
			log_prob = old_dist.log_prob(action_list[j][0][i])
			prob = np.exp(log_prob.detach().numpy())
			prob_old_traj.append(prob)

		# multiply along all
		prob_old_tau = np.prod(prob_old_traj)

		weight = prob_old_tau / (prob_tau + 1e-8)
		weight_list.append(np.max((min_isw,weight)))
		num_list.append(prob_old_tau)
		denom_list.append(prob_tau)
	
	return weight_list, num_list, denom_list

def compute_u(policy, optimizer, prev_u, isw, prev_g, beta):

	eps = np.finfo(np.float32).eps.item()
	R = 0
	policy_loss = []
	returns = []
	for r in policy.rewards[::-1]:
		R = r + args.gamma * R
		returns.insert(0, R)

	returns = torch.tensor(returns)
	returns = (returns - returns.mean()) / (returns.std() + eps)
	for log_prob, R in zip(policy.saved_log_probs, returns):
		policy_loss.append(-log_prob * R)
	optimizer.zero_grad() 
	policy_loss = torch.stack(policy_loss).sum() 
	print('Loss:',policy_loss)
	policy_loss.backward()

	# # list of shapes [torch.Size([64, dim]), torch.Size([64]), torch.Size([64, 64]), torch.Size([64]), torch.Size([3, 64]), torch.Size([3])
	# grad_shapes = [p.shape if p.requires_grad is True else None
	# 			for group in optimizer.param_groups for p in group['params']]
	# # list of num_params [128, 64, 4096, 64, 192, 3]   
	# grad_numel = [p.numel() if p.requires_grad is True else 0
	# 			for group in optimizer.param_groups for p in group['params']]
				  
	# #list of device [device(type='cuda', index=0), device(type='cuda', index=0) etc]
	# devices = [p.device for group in optimizer.param_groups for p in group['params']]

	# list of tensors gradients, each tensor has shape
	grad = [p.grad.detach().clone().flatten() if (p.requires_grad is True and p.grad is not None)
			else None for group in optimizer.param_groups for p in group['params']]
	
	# grad, prev_u and prev_g are all flatten grads
	assert grad[0].shape == prev_u[0].shape == prev_g[0].shape
	# list of flatten surrogate grads [128, 64, 4096, ...]
	grad_surrogate = [1]*len(grad)
	#u = beta*grad + (1-beta)*(prev_u + grad - isw*prev_g)
	for i in range(len(grad_surrogate)):
		grad_surrogate[i] = beta*grad[i] + (1-beta)*(prev_u[i] + grad[i] - isw*prev_g[i])
	return grad_surrogate

def main():
	# initialize env
	num_agents = args.num_agents
	dimension = args.dim
	fpath = os.path.join('mdpg_results_min_' + str(args.min_isw) + 'isw', args.env, str(dimension) + 'D', args.opt + 'beta='+ str(args.beta))

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
	if args.opt == 'adam':
		for i in range(num_agents):
			agents.append(Policy(state_dim=dimension).to(device))
			optimizers.append(optim.Adam(agents[i].parameters(), lr=3e-4))
	elif args.opt == 'sgd':
		for i in range(num_agents):
			agents.append(Policy(state_dim=dimension).to(device))
			optimizers.append(optim.SGD(agents[i].parameters(), lr=3e-4, momentum=args.momentum))
	elif args.opt == 'sgd_m':
		for i in range(num_agents):
			agents.append(Policy(state_dim=dimension).to(device))
			optimizers.append(SGD_M(agents[i].parameters(), lr=3e-4, momentum=args.momentum))
	elif args.opt == 'rmsprop':
		for i in range(num_agents):
			agents.append(Policy(state_dim=dimension).to(device))
			optimizers.append(optim.RMSprop(agents[i].parameters(), lr=3e-4))

	#initialization
	old_agents = copy.deepcopy(agents)
	print('Sampling initial trajectory')
	state = env.reset()
	R = 0
	for t in range(1, args.max_eps_len):  
		actions = []
		for policy in agents:
			action = select_action(state, policy)
			actions.append(action)
		actions = torch.as_tensor([actions])

		state, rewards, done = env.step(actions)

		for i in range(len(agents)):
			agents[i].rewards.append(rewards[i])
			
		state = torch.FloatTensor(state).to(device)
		R += np.sum(rewards)
		reset = t == args.max_eps_len-1
		if done or reset:
			print('Initial Trajectory: Reward', R, 'Done', done)
			R = 0
			break

	# initializating without consensus of u
	prev_u_list = []
	for policy, optimizer in zip(agents, optimizers):
		grads = compute_grads(policy, optimizer)
		prev_u_list.append(grads)
	agents = global_average(agents, num_agents)
	for policy, optimizer in zip(agents, optimizers):
		update_weights(policy, optimizer)

	# RL setup
	done = False
	R = 0 
	R_hist = []
	R_hist_plot = []
	isw_plot = []
	num_plot = []
	denom_plot = []

	action_list = []
	state_list = []
	
	for episode in range(args.num_episodes):
		state = env.reset()
		state_list.append(state)
		state = torch.FloatTensor(state).to(device)

		# phi is now old agent
		phi = copy.deepcopy(old_agents)
		# old_agent is now updated agent
		old_agent = copy.deepcopy(agents)

		if episode == args.num_episodes - 1:
			path = [state]

		# sample one trajectory
		for t in range(1, args.max_eps_len):  
			actions = []
			for policy in agents:
				action = select_action(state, policy)
				actions.append(action)
			actions = torch.as_tensor([actions])

			#step through enviroment with set of actions. rewards is list of reward
			state, rewards, done = env.step(actions)
			#print('State:', state, 'Action:', actions, 'Rewards:', rewards)
			action_list.append(actions)
			state_list.append(state)
			for i in range(len(agents)):
				agents[i].rewards.append(rewards[i])
	
			state = torch.FloatTensor(state).to(device)
			R += np.sum(rewards)
			reset = t == args.max_eps_len-1
			if done or reset:
				print(f'Done: {done} Reset:{reset} State:{state} reward:{rewards}')
				R_hist.append(R)
				R = 0
				break

		# compute ISW using latest traj with current agent and old agents
		isw_list, num, denom = compute_IS_weight(action_list, state_list, agents, phi, args.min_isw)
		print(isw_list)
		isw_plot.append(isw_list)
		num_plot.append(num)
		denom_plot.append(denom)

		# compute gradient of current trajectory using old agents. This requires old agents with gradients
		list_grad_traj_prev_weights = []
		for policy, old_policy, optimizer in zip(agents, phi, optimizers):
			prev_g = compute_grad_traj_prev_weights(state_list, action_list, policy, old_policy, optimizer)
			list_grad_traj_prev_weights.append(prev_g)

		grad_surrogate_list = []
		# compute gradient surrogate
		for policy, optimizer, prev_u, isw, prev_g in zip(agents, optimizers, prev_u_list, isw_list, list_grad_traj_prev_weights):
			grad_surrogate = compute_u(policy, optimizer, prev_u, isw, prev_g, args.beta)
			grad_surrogate_list.append(grad_surrogate)

		# take consensus of parameters
		agents = global_average(agents, num_agents)

		# update_weights with grad surrogate
		for policy, optimizer, grad_surrogate in zip(agents, optimizers, grad_surrogate_list):
			update_weights(policy, optimizer, grads=grad_surrogate)

		prev_u_list = copy.deepcopy(grad_surrogate_list)

		#update old_agents to current agent
		action_list = []
		state_list = []

		if episode % args.log_interval == 0:
			avg_reward = np.sum(R_hist)/len(R_hist)
			R_hist_plot.append(avg_reward)
			R_hist = []
			print(f'Episode:{episode} Average reward:{avg_reward:.2f}')
							
		if episode % 100 == 0:
			print(f'Last Action: {actions} State: {state} Reward: {rewards} Done: {done}')

	plt.figure()
	plt.plot(R_hist_plot)
	plt.ylabel('Reward')
	plt.xlabel('Episodes')
	plt.title(str(dimension) + '-d ' + ' ' + args.env + args.opt + ' ' + str(args.seed))
	plt.savefig(os.path.join(fpath, str(args.seed) + '_R.jpg'))
	plt.close()

	isw_plot = np.array(isw_plot)
	num_plot = np.array(num_plot)
	denom_plot = np.array(denom_plot)

	plt.figure()
	for i in range(num_agents):
		plt.plot(isw_plot[:,i], label='Agent' + str(i))
	plt.legend()
	plt.ylabel('Importance Sampling Weight')
	plt.xlabel('Episodes')
	plt.title(str(dimension) + '-d ' + ' ' + args.env + args.opt + ' ' + str(args.seed))
	plt.savefig(os.path.join(fpath, str(args.seed) + '_ISW.jpg'))

	plt.figure()
	for i in range(num_agents):
		plt.plot(num_plot[:,i], label='Agent' + str(i))
	plt.legend()
	plt.ylabel('Importance Sampling Weight (Numerator)')
	plt.xlabel('Episodes')
	plt.title(str(dimension) + '-d ' + ' ' + args.env + args.opt + ' ' + str(args.seed))
	plt.savefig(os.path.join(fpath, str(args.seed) + '_ISW_num.jpg'))

	plt.figure()
	for i in range(num_agents):
		plt.plot(denom_plot[:,i], label='Agent' + str(i))
	plt.legend()
	plt.ylabel('Importance Sampling Weight (Denominator)')
	plt.xlabel('Episodes')
	plt.title(str(dimension) + '-d ' + ' ' + args.env + args.opt + ' ' + str(args.seed))
	plt.savefig(os.path.join(fpath, str(args.seed) + '_ISW_denom.jpg'))

	np.save(os.path.join(os.path.join(fpath, 'R_array_' + str(args.seed) + '.npy')), R_hist_plot)

if __name__ == '__main__':
	main()