import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class Styblinski_Tang(gym.Env):
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
        self.prev_y = 1 #dummy variable
        self.action_space = spaces.Discrete(self.dimension)
        self.observation_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(self.dimension,), dtype=np.float32)
        self.done = False
        self.reward = 0
        self.reset()
        # only used for contour plotting purposes for 2D
        self.minima = np.array([-2.903534,-2.903534])*2
        
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
        # if np.less_equal((np.absolute(self.state - np.ones(self.dimension)*(-2.903534))), np.ones(self.dimension)*(1e-3)).all(): 
        #     self.done = True
        #     self.reward = 10
        return self.state, self.reward, self.done, self.y

    def reset(self):
        self.state =  np.random.uniform(low=-5, high=5, size=(self.dimension,))
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
        #https://www.sfu.ca/~ssurjano/stybtang.html
        summation = 0
        for i in range(self.dimension):
            summation = summation + self.state[i]**4 - 16*self.state[i]**2 + 5*self.state[i]
        y = 0.5*summation    
        return y

    def plot_eval_func(self, state):
        assert len(state) == 2, "state dimension surpasses 3D for visualization purposes"
        x, y = state
        return 0.5*(x**4 - 16*(x**2) + 5*x + y**4 - 16*(y**2) + 5*y)
