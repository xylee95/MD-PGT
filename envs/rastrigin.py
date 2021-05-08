import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class Rastrigin(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dimension=2, seed=0):
        #dimension of benchmark Rastrigin function
        #The function is usually evaluated on the hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, d. 
        self.seed = seed
        np.random.seed(self.seed)
        self.dimension = dimension
        self.max_bound = 5
        self.min_bound = -5
        self.y = 1 #dummy variable
        self.prev_y = 1 #dummy
        self.action_space = spaces.Discrete(self.dimension)
        self.observation_space = spaces.Box(low=self.min_bound, high=self.max_bound, shape=(self.dimension,), dtype=np.float32)
        self.done = False
        self.reward = 0
        self.step_size = 0.25
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
                change = 0.00
            delta.append(change)
        self.state = self.state + delta
        self.state = np.clip(self.state, self.min_bound, self.max_bound)
        self.prev_y = self.y
        self.y = self.eval_func(action)
        self.reward = self.get_reward()
        if np.less_equal((np.absolute(self.state)), np.ones(self.dimension)*1e-3).all():
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

        #reward = -0.1*self.y + (self.prev_y - self.y) - 0.1
        #reward = -0.1*self.y + 0.1*(self.prev_y - self.y)

        #r3
        reward = -0.1*self.y - 0.1

        #r4
        #reward = -0.1*np.linalg.norm(self.state)
        return reward
        
    def eval_func(self, action):
        assert len(action)==self.dimension
        # https://www.sfu.ca/~ssurjano/rastr.html
        sum_term = 0
        A = 10
        for i in range(self.dimension):
            sum_term = sum_term + (self.state[i]**2 - 10*np.cos(2*np.pi*self.state[i]))
        y = A*self.dimension + sum_term
        return y

    def plot_eval_func(self, state):
        assert len(state) == 2, "action dimension surpasses 3D for visualization purposes"
        x, y = state
        return 10*2 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))