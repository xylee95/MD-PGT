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
        self.max_bound = 5
        self.min_bound = -5
        self.y = 1 #dummy variable
        self.prev_y = 1 #dummy
        self.action_space = spaces.Discrete(self.dimension) #3 action per agent (up, down, stay) ^ num agents
        self.observation_space = spaces.Box(low=self.min_bound, high=self.max_bound, shape=(self.dimension,), dtype=np.float32)
        self.done = False
        self.reward = 0
        self.step_size = 0.5
        self.bounds = np.arange(self.min_bound, self.max_bound, self.step_size)
        self.reset()
        # only used for contour plotting purposes for 2D
        self.minima = np.array([0,0])
        
    def step(self, action):
        #action is vector of delta
        action = action[0].detach().numpy()
        delta = []
        for item in action:
            if item == 0:
                change = self.step_size
            elif item == 1:
                change = -1*self.step_size
            elif item == 2:
                change = 0
            delta.append(change)
        self.state = self.state + delta
        self.state = np.clip(self.state, self.min_bound, self.max_bound)
        self.prev_y = self.y
        self.y = self.eval_func(action)
        self.reward = self.get_reward()
        # optima at x = -y -z -a -b etc...
        if np.less_equal((np.absolute(self.state)), np.ones(self.dimension)*1e-5).all():
            self.done = True
            self.reward = 1
        return self.state, self.reward, self.done, self.y

    def reset(self):
        #self.state = np.floor(np.random.uniform(low=self.min_bound, high=self.max_bound, size=(self.dimension,)))
        self.state = np.random.choice(self.bounds, self.dimension, replace=True)
        while np.linalg.norm(self.state) < 5*self.step_size: #np.linalg.norm(self.max_bound):
            #self.state = np.floor(np.random.uniform(low=self.min_bound, high=self.max_bound, size=(self.dimension,)))
            #self.state = np.clip(self.state, self.min_bound, self.max_bound)
            self.state = np.random.choice(self.bounds, self.dimension, replace=True)

        print('Reset state:', self.state)
        self.done = False
        return np.array(self.state)


    def get_reward(self):
        # Reward compute based on y: we want previous y to be bigger than current y
        # Might also want to consider based on distance of x from optimal

        reward = -0.1*self.y + (self.prev_y - self.y) - 0.1
        #reward = -0.1*self.y + 0.1*(self.prev_y - self.y)

        #r3
        ##reward = -0.1*self.y 

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

class Quadratic5D(Quadratic):
    """docstring for Quadratic3D"""
    def __init__(self, dimension=5, seed=0):
        super(Quadratic5D, self).__init__(dimension=5)

    def eval_func(self, action):
        # optimum at x = -y-z
        assert len(action)==self.dimension
    
        A = np.ones((5,5))
        y = np.matmul(np.matmul(self.state, A), np.transpose(self.state))
        return y

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

