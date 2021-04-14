import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class Quadratic(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dimension=2, seed=0):
        self.seed = seed
        np.random.seed(self.seed)
        self.dimension = dimension
        self.min_action = -0.5
        self.max_action = 0.5 
        self.initial_pos =  np.random.uniform(low=-10, high=10, size=(self.dimension,))
        self.max_bound = 10
        self.min_bound = -10
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
        self.state = np.random.uniform(low=self.min_bound, high=self.max_bound, size=(self.dimension,))
        self.done = False
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
        y = self.state[0]**2 + 2*self.state[0]*self.state[1] + self.state[1]**2
        return y

    def plot_eval_func(self, state):
        x, y = state
        return x**2 + 2*x*y + y**2

class Quadratic3D(Quadratic):
    """docstring for Quadratic3D"""
    def __init__(self, dimension=3, seed=0):
        super(Quadratic3D, self).__init__(dimension=3)

    def eval_func(self, action):
        # optimum at x = -y-z
        assert len(action)==self.dimension
        y = self.state[0]**2 + self.state[1]**2 + self.state[2]**2 + \
            2*self.state[0]*self.state[1] + 2*self.state[0]*self.state[2] + 2*self.state[1]*self.state[2]
        return y

    def plot_eval_func(self, state):
        x, y, z = state
        return x**2 + y**2 + z**2 + 2*x*y + 2*x*z + 2*y*z

class Quadratic10D(Quadratic):
    """docstring for Quadratic3D"""
    def __init__(self, dimension=10, seed=0):
        super(Quadratic10D, self).__init__(dimension=10)

    def eval_func(self, action):
        # optimum at x = -y-z
        assert len(action)==self.dimension
    
        A = np.ones((10,10))
        y = np.matmul(np.matmul(self.state, A), np.transpose(self.state))
        return y

