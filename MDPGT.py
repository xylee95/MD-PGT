import torch
from torch.optim import Optimizer
from copy import copy, deepcopy

class MDPGT(Optimizer):
	def __init__(self, params, pi, lr=0.01, beta=0.95):
		if not 0.0 <= lr:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if not 0.0 < beta:
			raise ValueError("Invalid beta value: {}".format(beta))

		defaults = dict(pi=pi, lr=lr, beta=beta)

		super(MDPGT, self).__init__(params, defaults)

		self.prev_uk = []
		self.prev_grad = []
		self.grad_w_prev_param = []


	def __setstate__(self, state):
		super(MDPGT, self).__setstate__(state)

	@torch.no_grad()
	def step(self, closure=None):
		"""Performs a single optimization step.
		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
		"""
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for i, group in enumerate(self.param_groups):
			pi = group['pi']
			lr = group['lr']
			beta = group['beta']
			# beta = group['omega']
			omega = 1.0 # Temporary omega

			for j, p in enumerate(group['params']):
				if p.grad is None:
					continue
				d_p = p.grad.data

				# Compute SARAH with Importance Sampling Weight(ISW)
				SwISW = self.prev_uk[j] + d_p - omega*self.grad_w_prev_param[j]

				# Compute uk
				uk = beta*d_p + (1-beta)*SwISW

				# Update local policy gradient tracker (Gradient Tracking)
				new_vk = torch.zeros(d_p.size(),dtype=d_p.dtype)
				for pival in pi:
					pass


				# Update local estimate of the policy network parameter
				p.data.mul_(0.0) # zero out curent param
				# new_vk = torch.zeros(d_p.size(),dtype=d_p.dtype)
				for ag in range(len(pi)):
					tmp = p.add(d_p, alpha=-lr)
					p.add_(tmp)






		return loss

