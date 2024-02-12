import numpy as np
import matplotlib.pyplot as plt
from mountaincar import MountainCar
from tqdm import trange
import gym
np.random.seed(0)
import itertools

class Sarsa_Lambda:
    def __init__(self, env, epsilon, gamma, lam, max_steps=200, alpha=0.01):
        self.env = env
        self.degree = 3
        self.state = np.array([0,0])
        self.fourier_state = self.fourier_series(self.state)
        self.w = np.random.randn(np.shape(self.fourier_state)[0], len(self.env.actions))
        self.e = np.zeros_like(self.w)
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha


    def fourier_series(self, state):
        s = np.array(state).copy()
        s = self.normalize(s)
        s = np.array([s]).T
        iter = itertools.product(range(self.degree + 1), repeat=len(s))
        c = np.array([list(map(int, x)) for x in iter])
        return np.cos(np.pi * c.dot(s))

    def reset(self,e = None, trace=None, w=None, alpha=None):
        if e is not None:
            self.epsilon = 0.1
        if alpha is not None:
            self.alpha = 0.01
        if w is not None:
            self.w = np.zeros([np.shape(self.fourier_state)[0], len(self.env.actions)])
        if trace is not None:
            self.e = np.zeros_like(self.w)
        self.env.reset()

    def normalize(self, state):
        p, v = list(state)
        return (p + 1.2) / (0.6 + 1.2), (v + 0.07) / (2 * 0.07)

    def get_q_value(self, state, action):
        state_fourier = self.fourier_series(state)
        return np.dot(self.w[:, action].T, state_fourier)

    def decay_epsilon(self):
        if self.epsilon > 0.1:
            self.epsilon -= 0.1
        else:
            self.epsilon = 0.1
        if self.alpha > 0.01:
            self.alpha *= 0.999
        else:
            self.alpha = 0.001

    def epsilon_greedy(self, state):

        n_actions = len(self.env.actions)
        if not np.random.rand() < self.epsilon:
            values = np.array([self.get_q_value(state, action_) for action_ in range(n_actions)])
            action = np.argmax(values)
            # print('greedy')
        else:
            action = np.random.randint(n_actions)
            # print('explore')
        return action


    def run_episode(self):
        n_steps = np.zeros([500])
        self.reset(trace=True, w=True)
        for ep in trange(500):
            self.reset(trace=True)
            n_steps[ep] = self.run(ep)
            if ep % 200 == 0:
                self.decay_epsilon()
        return n_steps


    def run(self, ep):
        s = self.env.reset()
        a = self.epsilon_greedy(s)

        n_steps = 0
        total_reward = 0
        Q_old = 0
        while True:
            n_steps += 1

            s_prime, term = self.env.transition(s, a, n_steps)
            a_prime = self.epsilon_greedy(s_prime)

            reward = self.env.reward(term)

            psi = self.fourier_series(s)
            psi_prime = self.fourier_series(s_prime)
            Q = np.dot(self.w[:, a].T,  psi)

            if term:
                Q_prime = 0
            else:
                Q_prime = np.dot(self.w[:, a_prime].T, psi_prime)
            total_reward += reward
            print(f'State = {s}, Action {a}, reward: {total_reward},'
                  f' next state: {s_prime}, next action: {a_prime}')

            target = reward + self.gamma * Q_prime
            delta = target - Q
            delta_q = Q - Q_old
            #e ← γλe + ψ − αγλ(e>ψ) ψ
            self.e[:, :] *= self.gamma * self.lam
            self.e[:, a] += psi[0] - self.alpha * self.gamma * self.lam * (np.dot(self.e[:, a].T, psi) * psi)[0]

            # θ ← θ + α(δ + Q − Qold) e − α(Q − Qold)ψ
            error = (delta + delta_q) * self.e[:, :] - (delta_q) * psi
            self.w[:, :] += self.alpha * error

            Q_old = Q_prime
            s = s_prime
            a = a_prime

            if term:
                print('term')
                self.reset(trace=True)
                break
            if self.max_steps <= n_steps:
                print('max')
                self.reset(trace=True)
                break

        return total_reward

    def draw_plot(self, trial_reward, filename='tests.png'):
        key_list = list(trial_reward.keys())
        colors = ['b', 'g', 'r', 'c', 'm']

        for i in range(0, len(key_list)):
            key = key_list[i]
            mean_rewards = trial_reward[key]
            std_dev_rewards = np.std(trial_reward[key], axis=0)

            x_values = np.arange(100)

            plt.plot(x_values, mean_rewards, color=colors[i], label=f'Hyperparam {i + 1}')
            plt.fill_between(x_values, mean_rewards - (std_dev_rewards),
                             mean_rewards + std_dev_rewards, color=colors[i], alpha=0.2,
                             label=f'Standard Deviation{i + 1}')

        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Mean and Standard Deviation of Rewards')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig(filename)

        plt.show()

if __name__ == '__main__':
    env = MountainCar()
    n_eps = 1000
    alpha = 0.5
    lambda_ = 0.4
    epsilon = 0
    gamma = 0.9
    max_steps = 1000
    '''
    hyperparam = {
        'alpha': [0.01, 0.02, 0.03, 0.04, 0.05],
        'gamma': [0.8, 0.85, 0.9, 0.99, 0.9],
        'lambda': [0.4, 0.4, 0.4, 0.4, 0.4]
    }
    '''
    hyperparam = {
        'alpha': [0.01],
        'gamma': [0.9],
        'lambda': [0.4]
    }

    reward_dict = {}
    reward_list = []
    avg_std_dev = []

    for param in range(len(hyperparam['alpha'])):
        alpha = hyperparam['alpha'][param]
        gamma = hyperparam['alpha'][param]
        lam = hyperparam['lambda'][param]
        true_online_sarsa_lambda = Sarsa_Lambda(env, epsilon, gamma, lam, max_steps=max_steps, alpha=alpha)
        ave_reward_list = []
        for ep in trange(n_eps):
            true_online_sarsa_lambda.reset(trace=True, w=True)
            total_reward = true_online_sarsa_lambda.run(ep)
            reward_list.append(total_reward)
            if ep % 10 == 0:
                ave_reward_list.append(np.mean(reward_list))
                reward_list = []
        reward_dict[param] = ave_reward_list

    true_online_sarsa_lambda.draw_plot(reward_dict, 'hypertuning.png')
