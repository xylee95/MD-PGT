import numpy as np
import gym
from gym import spaces
from typing import List, Tuple

TRAP = -1
EMPTY = 0
AGENT = 1
WALL = 2
GOAL = 100

class grid:

    def __init__(self, size: Tuple[int,int], agents: int, seed: int):
        assert len(size) == 2, "grid size must be 2D"
        self.grid_row = size[0]
        self.grid_col = size[1]
        self.agents = agents
        np.random.seed(seed)

        # Setup action mapper
        self.action_mapper = {0: 'up', 1: 'down', 2: 'left', 3: 'right', 4: 'no-op'}
        self.store_position = {}
        self.env_params()

    def env_params(self):
        self.grid = np.zeros([self.grid_row, self.grid_col])
        self.state = []
        self.goal = []
        for i in range(self.agents):
            self.state.append(tuple(np.random.randint((self.grid_row, self.grid_col))))
            self.goal.append(((self.grid_row-1)//(i+1), (self.grid_col-1)//(i+1)))
            self.check_spawn(i)
            self.grid[self.state[i]] = AGENT
            self.grid[self.goal[i]] = GOAL
        self.state = np.array(self.state).reshape(-1)

    def check_spawn(self, idx):
        condition = 1
        while condition:
            up = (self.goal[idx][0] - 1, self.goal[idx][1])
            down = (self.goal[idx][0] + 1, self.goal[idx][1])
            left = (self.goal[idx][0], self.goal[idx][1] - 1)
            right = (self.goal[idx][0], self.goal[idx][1] + 1)
            if np.array_equal(self.state[idx], self.goal[idx]):
                self.state[idx] = tuple(np.random.randint((self.grid_row, self.grid_col)))
                continue
            elif np.array_equal(self.state[idx], up):
                self.state[idx] = tuple(np.random.randint((self.grid_row, self.grid_col)))
                continue
            elif np.array_equal(self.state[idx], down):
                self.state[idx] = tuple(np.random.randint((self.grid_row, self.grid_col)))
                continue
            elif np.array_equal(self.state[idx], left):
                self.state[idx] = tuple(np.random.randint((self.grid_row, self.grid_col)))
                continue
            elif np.array_equal(self.state[idx], right):
                self.state[idx] = tuple(np.random.randint((self.grid_row, self.grid_col)))
                continue
            else:
                condition = 0

    def reset(self):
        self.env_params()
        return self.state

    def getreward(self):
        # return -np.sqrt((self.goal[0] - self.state[0])**2 + (self.goal[1] - self.state[1])**2)
        # return self.grid[self.state]
        reward = 0

        for i in range(self.agents):
            if np.array_equal(self.goal[i], self.state[i]):
                reward += 1
            else:
                reward += 0

            # collision check
            keys = list(self.store_position.keys())
            keys.remove(i)
            for k in keys:
                if np.array_equal(self.state[i], self.store_position[k]):
                    reward -= 2
                else:
                    reward += 0

        # if each agents reach their respective goals, gain huge reward
        if reward == self.agents:
            reward += 100*self.agents

        return reward

    def getdone(self):
        done = []
        for i in range(self.agents):
            if np.array_equal(self.goal[i], self.state[i]):
                done.append(1)
            else:
                done.append(0)
        
        if np.sum(done) == self.agents:
            done = 1
        else:
            done = 0

        return done

    def step(self, raw_action):

        assert len(raw_action) == self.agents
        self.state = self.state.reshape(self.agents, -1)

        """
        Reference
        0 | 1 | 2
        --
        1
        --
        2
        --

        (row, col)

        ------------
        Actions
        UP    -> row -1
        DOWN  -> row +1
        LEFT  -> col -1
        RIGHT -> col +1
        STAY  -> no-op
        ------------
        """

        for i in range(self.agents):
            self.grid[tuple(self.state[i])] = EMPTY
            self.grid[tuple(self.goal[i])] = GOAL
            action = self.action_mapper[raw_action[i].numpy()[0]]

            if action == 'up':
                nextstep =  tuple(np.array([self.state[i][0] - 1, self.state[i][1]]))
            elif action == 'down':
                nextstep =  tuple(np.array([self.state[i][0] + 1, self.state[i][1]]))
            elif action == 'left':
                nextstep =  tuple(np.array([self.state[i][0], self.state[i][1] - 1]))
            elif action == 'right':
                nextstep =  tuple(np.array([self.state[i][0], self.state[i][1] + 1]))
            else:
                nextstep = tuple(np.array([self.state[i][0], self.state[i][1]]))

            if nextstep[0] >= 0 and nextstep[0] <= self.grid_row - 1:
                if nextstep[1] >= 0 and nextstep[1] <= self.grid_col - 1:
                    self.state[i] = nextstep

            self.grid[tuple(self.state[i])] = AGENT
            self.store_position[i] = self.state[i]
        reward = self.getreward()
        done = self.getdone()
        self.state = self.state.reshape(-1)
        self.store_position = {}
        return self.state, reward, done, 0

    def render(self):

        for i in range(self.grid_row):
            rowstrings = '--'
            for rs in range(self.grid_col):
                rowstrings += '-'*4
            print(rowstrings)
            out = '| '
            for j in range(self.grid_col):
                if self.grid[i, j] == AGENT:
                    token = '*'
                elif self.grid[i, j] == TRAP:
                    token = 'T'
                elif self.grid[i, j] == EMPTY:
                    token = '0'
                elif self.grid[i, j] == GOAL:
                    token = 'G'
                elif self.grid[i, j] == WALL:
                    token = '^'
                else:
                    print(f"Invalid value inside grid: {self.grid[i, j]}")
                    print(self.grid[i, j] == AGENT)
                    print(self.grid)
                out += token + ' | '
            print(out)
        print(rowstrings)

