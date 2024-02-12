import numpy as np
import math

np.random.seed(0)
import itertools

class MountainCar:
    def __init__(self):
        self.state = np.zeros([1, 2])
        self.actions = [0, 1, 2]
        self.gravity = 0.0025
        self.force = 0.001
        self.degree = 3

    def reset(self):
        return self.d_0()


    def d_0(self):
        self.state = np.array([np.random.uniform(-0.6, -0.4), 0])
        return self.state

    def transition(self, state, action, len_ep):
        position, velocity = state
        velocity_new = velocity + (action - 1) * self.force - np.cos(3 * position) * self.gravity
        position_new = position + velocity_new
        if position_new < -1.2:
            position_new = -1.2
            velocity_new = 0
        if position_new > 0.5:
            position_new = 0.5
            velocity_new = 0

        s_prime = np.array([position_new, velocity_new])

        if self.terminal(position_new, len_ep):
            term = True
        else:
            term = False

        return s_prime, term

    def reward(self, ep):
        if ep:
            return 0
        else:
            return -1



    def terminal(self, position=None, length=None):
        if position >= 0.5 or length == 200:
            return True
        else:
            return False
