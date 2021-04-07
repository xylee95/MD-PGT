import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from numpy import linalg as LA

class Sphere(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dimension=2, seed=0):
        #dimension of benchmark Rastrigin function
        #The function is usually evaluated on the hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, d. 
        self.seed = seed
        np.random.seed(self.seed)
        self.dimension = dimension
        self.min_action = -0.5
        self.max_action = 0.5 
        self.initial_pos =  np.random.uniform(low=-10, high=10, size=(self.dimension,))
        self.max_bound = 5
        self.min_bound = -5
        self.y = 1 #dummy variable
        self.prev_y = 1 #dummy
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(self.dimension,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(self.dimension,), dtype=np.float32)
        self.done = False
        self.reward = 0
        self.reset()
        
    def step(self, action):
        #action is vector of delta
        action = action[0].detach().numpy()
        self.state = self.state + action
        self.state = np.clip(self.state, self.min_bound, self.max_bound)
        self.prev_y = self.y
        self.y = self.eval_func(action)
        self.reward = self.get_reward()
        if np.less_equal((np.absolute(self.state)), np.ones(self.dimension)*1e-3).all():
            self.done = True
            self.reward = 10
        # another termintion condition on f(y) maybe?
        return self.state, self.reward, self.done, self.y

    def reset(self):
        self.state =  np.random.uniform(low=self.min_bound, high=self.max_bound, size=(self.dimension,))
        return np.array(self.state)

    def get_reward(self):
        # Reward compute based on y: we want previous y to be bigger than current y
        # Might also want to consider based on distance of x from optimal

        #reward = -0.1*self.y + (self.prev_y - self.y)
        #reward = -0.1*self.y + 0.1*(self.prev_y - self.y)

        #r3
        reward = -0.1*self.y 

        #r4
        #reward = -0.1*np.linalg.norm(self.state)
        return reward
        
    def eval_func(self, action):
        # optimum at x = y = 0
        assert len(action)==self.dimension
        y = 0
        for i in range(self.dimension):
            y += self.state[i]**2
        return y

    def plot_eval_func(self, action):
        assert len(action) == 2, "action dimension surpasses 3D for visualization purposes"
        x, y = action
        return x**2 + y**2




