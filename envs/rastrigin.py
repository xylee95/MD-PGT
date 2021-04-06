import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from numpy import linalg as LA

class Rastrigin(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dimension=2):
        #dimension of benchmark Rastrigin function
        #The function is usually evaluated on the hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, d. 
        self.dimension = dimension
        self.min_action = -0.5
        self.max_action = 0.5 
        self.initial_pos =  np.random.uniform(low=-5.12, high=5.12, size=(self.dimension,))
        self.max_bound = 5.12
        self.min_bound = -5.12
        self.goal_position = 0
        self.y = 1 #dummy variable
        self.prev_y = 1 #dummy
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(self.dimension,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(self.dimension,), dtype=np.float32)
        #self.seed()
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
        if np.less_equal( (np.absolute(self.state)), np.ones(self.dimension)*1e-3).all():
            self.done = True
            self.reward = 10
        # another termintion condition on f(y) maybe?
        return self.state, self.reward, self.done, self.y

    def reset(self):
        self.state =  np.random.uniform(low=-5.12, high=5.12, size=(self.dimension,))
        return np.array(self.state)

    def get_reward(self):
        # Reward compute based on y: we want previous y to be bigger than current y
        # Might also want to consider based on distance of x from optimal

        #reward = -0.1*self.y + (self.prev_y - self.y)
        #reward = -0.1*self.y + 0.1*(self.prev_y - self.y)

        #r3
        #reward = -0.1*self.y 

        #r4
        reward = -0.1*np.linalg.norm(self.state)
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

    def plot_eval_func(self, action):
        assert len(action) == 2, "action dimension surpasses 3D for visualization purposes"
        x, y = action
        return 10*2 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))

class Rastrigin2D(Rastrigin):
    """docstring for Rastrigin2D"""
    def __init__(self):
        super(Rastrigin2D, self).__init__(dimension=2)


class Rastrigin5D(Rastrigin):
    """docstring for Rastrigin5D"""
    def __init__(self):
        super(Rastrigin5D, self).__init__(dimension=5)


class Rastrigin10D(Rastrigin):
    """docstring for Rastrigin10D"""
    def __init__(self):
        super(Rastrigin10D, self).__init__(dimension=10)

class Rastrigin20D(Rastrigin):
    """docstring for Rastrigin20D"""
    def __init__(self):
        super(Rastrigin20D, self).__init__(dimension=20)

