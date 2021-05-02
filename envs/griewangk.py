import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class Griewangk(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dimension=2, seed=0):
        #dimension of benchmark Griewangk function
        #The function is usually evaluated on the hypercube xi ∈ [-600, 600], for all i = 1, …, d. 
        self.seed = seed
        np.random.seed(self.seed)
        self.dimension = dimension
        self.max_bound = 25
        self.min_bound = -25
        self.y = 1 #dummy variable
        self.prev_y = 1 #dummy variable
        self.action_space = spaces.Discrete(self.dimension) #3 action per agent (up, down, stay) ^ num agents
        self.observation_space = spaces.Box(low=self.min_bound, high=self.max_bound, shape=(self.dimension,), dtype=np.float32)
        self.reward = 0
        self.reset()
        # only used for contour plotting purposes for 2D
        self.minima = np.array([0,0])

    def step(self, action):
        #action is vector of delta
        action = action[0].detach().numpy()
        delta = []
        for item in action:
            if item == 0:
                change = 0.05
            elif item == 1:
                change = -0.05
            elif item == 2:
                change = 0.00
            delta.append(change)
        self.state = self.state + delta
        self.state = np.clip(self.state, self.min_bound, self.max_bound)
        self.prev_y = self.y
        self.y = self.eval_func(action)
        self.reward = self.get_reward()
        # if np.less_equal((np.absolute(self.state)), np.ones(self.dimension)*1e-3).all():
        #     self.done = True
        #     self.reward = 10
        return self.state, self.reward, self.done, self.y

    def reset(self):
        self.state =  np.random.uniform(low=self.min_bound, high=self.max_bound, size=(self.dimension,))
        self.done = False
        return np.array(self.state)

    def get_reward(self):
        # Reward compute based on y: we want previous y to be bigger than current y
        # Might also want to consider based on distance of x from optimal

        reward = -0.1*self.y + (self.prev_y - self.y)
        #reward = -0.1*self.y + 0.1*(self.prev_y - self.y)

        #r3
        #reward = -0.1*self.y 

        #r4
        #reward = -0.1*np.linalg.norm(self.state)
        return reward
        
    def eval_func(self, action):
        assert len(action)==self.dimension
        # https://www.sfu.ca/~ssurjano/Griewangk.html
        term1 = 0
        term2 = 0

        for i in range(self.dimension):
            term1 = term1 + (self.state[i]**2)/4000
            term2 = term2*np.cos(self.state[i]/np.sqrt(i+1))

        y = term1 - term2 + 1
        return y

    def plot_eval_func(self, state):
        assert len(state) == 2, "action dimension surpasses 3D for visualization purposes"
        x, y = state
        return ((x**2 + y**2)/4000) - np.cos(x)*np.cos(y/np.sqrt(2)) + 1
