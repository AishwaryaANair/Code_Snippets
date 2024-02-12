import numpy as np
import random
import copy


class Gridworld:
    def __init__(self, terminal=None):
        if terminal is None:
            terminal = []
        self.actions = [0, 1, 2, 3]
        self.q = np.zeros((5, 5, 4))
        self.q[4, 4, :] = 0
        self.v = np.zeros((5, 5))
        self.obstacles = [(2, 2), (3, 2)]
        self.water = [(4, 2)]
        self.goal = [(4, 4)]
        self.terminal = self.goal + terminal



    def d_0(self):
        while True:
            s = (np.random.randint(0, 5), np.random.randint(0, 5))
            if s not in self.obstacles and s not in self.terminal:
                break
        #print(s)
        return s

    def reset(self):
        self.v = np.zeros([5, 5])

    def reward(self, s):
        if s in self.goal:
            return 10
        if s in self.water:
            return -10
        return 0

    def transition(self, s, a):
        s_prime = self.next_state(s, a)
        #print(s_prime)
        next_x, next_y = s_prime
        if next_x > 4 or next_y > 4 or next_x < 0 or next_y < 0:
            return s
        elif s_prime in self.obstacles:
            return s
        else:
            return s_prime

    def is_goal(self, s):
        if s in self.terminal:
            return True
        else:
            False

    def next_state(self, s, a):
        if self.is_goal(s):
            return 'term'

        next_state = list()
        r, c = s
        if a == 0:
            next_state = [(r, c),(r - 1, c), (r, c - 1), (r, c + 1)]
        if a == 1:
            next_state = [(r, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        if a == 2:
            next_state = [(r, c), (r, c + 1), (r-1, c), (r+1, c)]
        if a == 3:
            next_state = [(r, c), (r, c - 1), (r-1, c), (r+1, c)]

        return random.choices(next_state, [0.1, 0.8, 0.05, 0.05])[0]

    def print_matrix(self, val):
        print('Value function')
        for line in val:
            print(', '.join([f"{float(item):.4f}" if item else "0.0000" for item in line]))
