import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class LineWorld(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, dimension=2, seed=0):
		self.seed = seed
		np.random.seed(self.seed)
		self.dimension = dimension
		self.max_bound = 1
		self.min_bound = -1
		self.action_space = spaces.Discrete(self.dimension) #3 action per agent (up, down, stay) ^ num agents
		self.observation_space = spaces.Box(low=self.min_bound, high=self.max_bound, shape=(self.dimension,), dtype=np.float32)
		self.done = False
		self.reward = 0
		self.step_size = 0.1
		self.bounds = np.arange(self.min_bound, self.max_bound, self.step_size)
		self.reset()

	def step(self, action):
		#action is vector of decision (0: up, 1: down, 2: stay)
		action = action[0].detach().numpy()
		delta = []
		for item in action:
			if item == 0:
				change = self.step_size
			elif item == 1:
				change = -1*self.step_size
			elif item == 2:
				change = 0.00
			delta.append(change)
		self.state = self.state + delta
		self.state = np.clip(self.state, self.min_bound, self.max_bound)
		self.reward = self.get_reward()
		if np.less_equal((np.absolute(self.state)), np.ones(self.dimension)*1e-3).all():
			self.done = True
			self.reward = [1]*self.dimension
		return self.state, self.reward, self.done
		
	def reset(self):
		self.state = np.random.choice(self.bounds, self.dimension, replace=True)
		while np.linalg.norm(self.state) < 5*self.step_size:
			self.state = np.random.choice(self.bounds, self.dimension, replace=True)
		#print('Reset state:', self.state)
		self.done = False
		return np.array(self.state)

	def get_reward(self):
		reward = -0.1*np.abs(self.state)
		return reward




