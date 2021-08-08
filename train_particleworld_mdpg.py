import argparse
import gym
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pfrl
import pdb
import envs
from envs import make_particleworld
import copy
import model
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
parser.add_argument('--max_eps_len', type=int, default=50, help='Number of steps per episode')
parser.add_argument('--num_episodes', type=int, default=50000, help='Number training episodes')
parser.add_argument('--env', type=str, default='particle_world', help='Training env')
parser.add_argument('--gpu', type=bool, default=False, help='Enable GPU')
parser.add_argument('--opt', type=str, default='sgd_m', help='Optimizer')
parser.add_argument('--momentum', type=float, default=0.0, help='Momentum term for SGD')
parser.add_argument('--beta', type=float, default=0.9, help='Beta term for surrogate gradient')
parser.add_argument('--min_isw', type=float, default=0.0, help='Minimum value to set ISW')
parser.add_argument('--minibatch_init', type=bool, default=False, help='Initialize grad with minibatch')
parser.add_argument('--minibatch_size', type=int, default=32, help='Number of trajectory for warm startup')
parser.add_argument('--topology', type=str, default='dense', choices=('dense','ring','bipartite'))
args = parser.parse_args()
torch.manual_seed(args.seed)

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

def main():
	# initialize env
	num_agents = args.num_agents
	dimension = args.dim
	if args.minibatch_init == False:
		fpath = os.path.join('mdpg_results_min_global_' + str(args.min_isw) + 'isw', args.env, str(dimension) + 'D', args.opt + 'beta='+ str(args.beta) + '_' + args.topology)
	elif args.minibatch_init == True:
		fpath = os.path.join('mdpg_results_min_global_' + str(args.min_isw) + 'isw', args.env, str(dimension) + 'D', args.opt + 'beta='+ str(args.beta) + '_' + args.topology + 'MI' + str(args.minibatch_size))
	if not os.path.isdir(fpath):
		os.makedirs(fpath)

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
		optimizers.append(SGD_M(agents[i].parameters(), lr=3e-4, momentum=args.momentum))

	# load connectivity matrix
	pi = load_pi(num_agents=args.num_agents, topology=args.topology)
	if args.minibatch_init:
		# if using Minibatch - Initialization
		# For i in range B trajectories:
		#    Sample traj
		#    Compute grads
		#    Store grads
		#    Average grads
		old_agents = copy.deepcopy(agents)
		print('Sampling initial minibatch of trajectory')
		state = sample_env.reset()
		state = np.concatenate(state).ravel().tolist()

		R = 0
		minibatch_grads = []
		for i in range(args.minibatch_size):
			for t in range(1, args.max_eps_len):  
				actions = []
				for policy in agents:
					action = model.select_action(state, policy)
					actions.append(action)

				state, rewards, done_n, _ = sample_env.step(actions)
				state = np.concatenate(state).ravel().tolist()
				done = all(item == True for item in done_n)

				for j in range(len(agents)):
					agents[j].rewards.append(rewards[j])
					
				R += np.sum(rewards)
				reset = t == args.max_eps_len-1
				if done or reset:
					print('Batch Initial Trajectory ' + str(i) + ': Reward', R, 'Done', done)
					R = 0
					break

			single_traj_grads = []
			for policy, optimizer in zip(agents, optimizers):
				grads = compute_grads(args, policy, optimizer, minibatch_init=True)
				single_traj_grads.append(grads) #list of num_agent x list grads of every layer
				optimizer.zero_grad()
				del policy.rewards[:]
				del policy.saved_log_probs[:]

			minibatch_grads.append(single_traj_grads) #list of minibatch x num_agent x list of grads of every layer

		# need grads to be shape num_agent x list of grads of every layer
		minibatch_grads = np.asarray(minibatch_grads)
		minibatch_grads = np.mean(minibatch_grads, 0) #average across batch
		minibatch_grads = minibatch_grads.tolist()
		# initializating with consensus of weights and grads
		prev_u_list = []
		for avg_grads in minibatch_grads:
			temp = []
			for layer in avg_grads:
				temp.append(torch.FloatTensor(layer))
			prev_u_list.append(temp)
	else:
		#initialization
		old_agents = copy.deepcopy(agents)
		print('Sampling initial trajectory')
		state = sample_env.reset()
		state = np.concatenate(state).ravel().tolist()
		R = 0
		for t in range(1, args.max_eps_len):  
			actions = []
			for policy in agents:
				action = model.select_action(state, policy)
				actions.append(action)

			state, rewards, done_n, _ = sample_env.step(actions)
			state = np.concatenate(state).ravel().tolist()
			done = all(item == True for item in done_n)

			for j in range(len(agents)):
				agents[j].rewards.append(rewards[j])

			R += np.sum(rewards)
			reset = t == args.max_eps_len-1
			if done or reset:
				print('Initial Trajectory: Reward', R, 'Done', done)
				R = 0
				break

		# initializating with consensus of weights and grads
		prev_u_list = []
		for policy, optimizer in zip(agents, optimizers):
			grads = compute_grads(args, policy, optimizer, minibatch_init=False)
			prev_u_list.append(grads)

	#agents = global_average(agents, num_agents)
	agents = take_param_consensus(agents, pi)
	
	for policy, optimizer, u_k in zip(agents, optimizers, prev_u_list):
		update_weights(policy, optimizer, grads=u_k)

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

	env = make_particleworld.make_env('simple_spread', num_agents=num_agents, num_landmarks=dimension)
	env.discrete_action_input = True #set action space to take in discrete numbers 0,1,2,3
	
	for episode in range(args.num_episodes):
		state = env.reset()
		state = np.concatenate(state).ravel().tolist()
		state_list.append(state)

		# phi is now old agent
		phi = copy.deepcopy(old_agents)
		# old_agent is now updated agent
		old_agent = copy.deepcopy(agents)

		# sample one trajectory
		for t in range(1, args.max_eps_len):  
			actions = []
			for policy in agents:
				action = model.select_action(state, policy)
				actions.append(action)

			state, rewards, done_n, _ = env.step(actions)
			state = np.concatenate(state).ravel().tolist()
			done = all(item == True for item in done_n)

			action_list.append(torch.as_tensor([actions]))
			state_list.append(state)
			for i in range(len(agents)):
				#print('r:', rewards[i])
				agents[i].rewards.append(rewards[i])
			
			R += sum(rewards)
			reset = t == args.max_eps_len-1
			if done or reset:
				#print(f'Eps: {episode} Done: {done_n} Reset:{reset} State:{state} reward:{rewards}')
				print(f'Eps: {episode} Done: {done_n} Reset:{reset} reward:{rewards}')
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
		old_agent_optimizers = []
		for i in range(num_agents):
			old_agent_optimizers.append(SGD_M(phi[i].parameters(), lr=3e-4, momentum=args.momentum))

		list_grad_traj_prev_weights = []
		for policy, old_policy, optimizer in zip(agents, phi, old_agent_optimizers):
			prev_g = compute_grad_traj_prev_weights(args, state_list, action_list, policy, old_policy, optimizer)
			list_grad_traj_prev_weights.append(prev_g)

		u_k_list = []
		# compute u_k
		for policy, optimizer, prev_u, isw, prev_g in zip(agents, optimizers, prev_u_list, isw_list, list_grad_traj_prev_weights):
			u_k = compute_u(args, policy, optimizer, prev_u, isw, prev_g, args.beta)
			u_k_list.append(u_k)

		# take consensus of parameters 
		# agents = global_average(agents, num_agents)
		agents = take_param_consensus(agents, pi)


		# update_weights with local grad surrogate, u_k
		for policy, optimizer, u_k in zip(agents, optimizers, u_k_list):
			update_weights(policy, optimizer, grads=u_k)

		prev_u_list = copy.deepcopy(u_k_list)

		#update old_agents to current agent
		action_list = []
		state_list = []

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