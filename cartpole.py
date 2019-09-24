import random
import warnings
from collections import deque

import gym
import numpy as np
from gym.envs.classic_control import CartPoleEnv
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor

from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001  # unused as we use Experience Replay type of Q-Learning
# See more on Experience Replay here: https://datascience.stackexchange.com/questions/20535/what-is-experience-replay-and-what-are-its-benefits

MEMORY_SIZE = 100000  # used only for Non-Incremental learning, i.e. partial_fit=False
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNSolver:

    def __init__(self, action_space, is_partial_fit: bool = False):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self._is_partial_fit = is_partial_fit

        if is_partial_fit:
            # Here you can use only Incremental Models: https://scikit-learn.org/0.18/modules/scaling_strategies.html
            regressor = SGDRegressor()
            self.model = MultiOutputRegressor(regressor)
        else:
            # Here you can use whatever regression model you want, simple or Incremental
            # The sklearn regression models can be found by searching for "regress" at https://scikit-learn.org/stable/modules/classes.html

            # Ex:
            #regressor = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
            #regressor = LGBMRegressor(n_estimators=100, n_jobs=-1)

            regressor = AdaBoostRegressor(n_estimators=10)
            self.model = MultiOutputRegressor(regressor)

        self.isFit = False

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        if self.isFit == True:
            q_values = self.model.predict(state)
        else:
            q_values = np.zeros(self.action_space).reshape(1, -1)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        X = []
        targets = []
        batch = random.sample(self.memory, BATCH_SIZE)
        if len(self.memory) % 1000 == 0 and len(self.memory)< MEMORY_SIZE:
            print(f"Memory size: {len(self.memory)}")
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if self.isFit:
                if not terminal:
                    q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
                q_values = self.model.predict(state)[0]
            else:
                q_values = np.zeros(self.action_space)
            q_values[action] = q_update

            if self._is_partial_fit:
                self.model.partial_fit([list(state[0])], [q_values])
            else:
                X.append(list(state[0]))
                targets.append(q_values)

        if not self._is_partial_fit:
            self.model.fit(X, targets)

        self.isFit = True
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cartpole():
    env: CartPoleEnv = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(action_space, is_partial_fit=False)
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            # comment next line for faster learning, without stopping to show the GUI
            env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cartpole()
