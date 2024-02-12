import numpy as np
import matplotlib.pyplot as plt
from gridworld import Gridworld
from tqdm import trange

np.random.seed(0)


class Sarsa_Gridworld:
    def __init__(self, env, epsilon, gamma, lam, num_steps, alpha=0.01):
        self.env = env
        self.num_steps = num_steps
        self.epsilon = epsilon
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        self.policy = [len(v) * [''] for v in self.env.v]
        self.pi = self.env.q.copy()
        self.e = np.zeros_like(self.env.q)


    def get_q_value(self, state, action):
        r, c = state
        return self.env.q[r, c, action]

    def reset(self, epsilon):
        self.epsilon = epsilon

    def update_q(self, state, action, reward, s_prime, a_prime):
        r, c = state
        r_prime, c_prime = s_prime
        if s_prime in self.env.terminal:
            next_q = 0
        else:
            next_q = self.env.q[r_prime, c_prime, a_prime]

        delta = (reward + self.gamma * next_q - self.env.q[r, c, action])
        self.env.q[r, c, action] = self.env.q[r, c, action] + self.alpha * delta
        return delta

    def calculate_values(self):
        for r in range(5):
            for c in range(5):
                max_act = np.argmax(self.env.q[r, c, :])
                max_ep = (1 - self.epsilon) + (self.epsilon / len(self.env.actions))
                ep = self.epsilon / len(self.env.actions)

                self.pi[r, c, :] = ep
                self.pi[r, c, max_act] = max_ep

                self.env.v[r, c] = np.dot(self.pi[r, c, :], self.env.q[r, c, :])

        return self.env.v

    def calculate_variance(self):
        self.calculate_values()
        optimal_v = np.array([[4.0187, 4.5548, 5.1575, 5.8336, 6.4553],
                              [4.3716, 5.0324, 5.8013, 6.6473, 7.3907],
                              [3.8672, 4.3900, 0.0000, 7.5769, 8.4637],
                              [3.4182, 3.8319, 0.0000, 8.5738, 9.6946],
                              [2.9977, 2.9309, 6.0733, 9.6946, 0.0000]])
        num_states = 22
        var = 0
        for r in range(5):
            for c in range(5):
                var+= np.square(self.env.v[r, c] - optimal_v[r, c])

        return var/num_states

    def epsilon_greedy(self, state):

        n_actions = len(self.env.actions)
        if not np.random.binomial(1, self.epsilon):
            values = np.array([self.get_q_value(state,
                                                action_) for action_ in range(n_actions)])
            action = np.argmax(values)
            # print('greedy')
        else:
            action = np.random.randint(n_actions)
            # print('explore')
        return action

    def optimal_policy(self):
        for r in range(5):
            for c in range(5):
                if (r, c) not in self.env.obstacles and (r, c) not in self.env.terminal:

                    max_a = np.argmax(self.env.q[r, c, :])
                    if max_a == 0:
                        self.policy[r][c] = '↑'
                    if max_a == 1:
                        self.policy[r][c] = '↓'
                    if max_a == 2:
                        self.policy[r][c] = '→'
                    if max_a == 3:
                        self.policy[r][c] = '←'
                if (r, c) in self.env.obstacles:
                    self.policy[r][c] = ' '
                if (r, c) in self.env.terminal:
                    self.policy[r][c] = 'G'

        self.print_matrix()

    def print_matrix(self):
        print('Policy')
        for line in self.policy:
            print(', '.join([f"{item}" if item else " " for item in line]))

    def decay_epsilon(self):
        self.epsilon -= 0.1

    def run_episode(self):
        n_steps = []
        variance = np.zeros([1000])
        for ep in trange(1000):
            n_steps.append(self.run(ep))
            variance[ep] = self.calculate_variance()
            if ep % 200 == 0:
                self.decay_epsilon()

        return n_steps, variance

    def run(self, ep):
        s = self.env.d_0()
        a = self.epsilon_greedy(s)
        n_steps = self.num_steps
        total_reward = 0
        while True:
            n_steps += 1

            s_prime = self.env.transition(s, a)
            a_prime = self.epsilon_greedy(s_prime)
            #print(s, a, self.env.reward(s_prime), s_prime, a_prime)
            total_reward += self.env.reward(s_prime)


            delta = self.update_q(s, a, self.env.reward(s_prime), s_prime, a_prime)

            self.e[s[0], s[1], a] += 1

            for r in range(5):
                for c in range(5):
                    if (r, c) not in self.env.obstacles or (r, c) not in self.env.terminal:
                        for a in self.env.actions:
                            self.env.q[r, c, a] += + self.alpha * delta * self.e[r, c, a]
                            self.e[r, c, a] = self.gamma * self.lam * self.e[r, c, a]

            s = s_prime
            a = a_prime

            if s_prime in self.env.terminal:
                return n_steps

    def plot(self, labels, path, num_steps=None, var=None):
        if num_steps is not None:
            num_steps = np.add.accumulate(num_steps)
            plt.plot(num_steps, np.arange(1, len(num_steps) + 1))
        if var is not None:
            plt.plot(np.arange(1, len(var) + 1), var)

        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

        plt.savefig(path)
        plt.show()
        plt.close()


if __name__ == '__main__':

    epsilon = 0.6
    gamma = 0.9
    lam = 0.4
    alpha = 0.1
    num_steps = 0

    env = Gridworld()
    gw_sarsa = Sarsa_Gridworld(env=env, epsilon=epsilon, gamma=gamma, lam=lam, alpha=alpha, num_steps=num_steps)

    n_steps_avg = []
    var_avg = np.zeros([20, 1000])
    for i in range(0, 20):
        gw_sarsa.reset(epsilon)
        n_steps, var = gw_sarsa.run_episode()
        n_steps_avg.append(n_steps)
        val = gw_sarsa.calculate_variance()
        var_avg[i] = var
        #print(gw_sarsa.env.q)
        gw_sarsa.optimal_policy()

    n_steps_avg = np.asarray(n_steps_avg)
    avg_steps = np.mean(n_steps_avg, axis=0)

    avg_var = np.mean(var_avg, axis=0, keepdims=True)
    #print(avg_var)

    gw_sarsa.plot(['Timesteps', 'Episodes'], './gridworld_sarsa_learn.png', num_steps=avg_steps)
    gw_sarsa.plot(['Episodes', 'Variance'], './gridworld_sarsa_mse.png', var=avg_var[0])
