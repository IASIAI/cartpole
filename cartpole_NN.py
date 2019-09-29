import random
from collections import deque
from typing import Tuple

import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from scores import score_logger

"""
Vanilla Multi Layer Perceptron version.
The below configuration starts converging to solution after ~7 runs with STOCHASTIC_TRAIN=True
"""

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 384
BATCH_SIZE = 20
Q_LEARN_RATE = 0.999

# Setting next parameter to True would force training of the network on the whole random-sample batch every step/action
# When on True - it converges faster to solution, with less runs, but consuming more resources
TRAIN_EACH_STEP = True

# Stochastic training - ie train with batches of size 1, thus offering faster feedback loop of predicted Q-values
# Convergence is significantly faster and starts converging after 7 runs
# As higher the BATCH_SIZE, as higher might be difference in conv speed offered by this, but slowing-down the training
STOCHASTIC_TRAIN = True

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.93
PLOT_REFRESH = 10


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.isFit = False

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, q_val, reward, next_state, done):
        self.memory.append((state, action, q_val, reward, next_state, done))

    def act(self, state) -> Tuple[int, float]:
        if np.random.rand() < self.exploration_rate:
            action = random.randrange(self.action_space)
            return action, 0.
        q_values = self.model.predict(state)
        return np.argmax(q_values[0]), np.max(q_values[0])

    def experience_replay_new(self):
        if self.isFit:
            self.exploration_rate *= EXPLORATION_DECAY
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        if not self.isFit:
            self.model.fit(batch[0][0], np.array([np.zeros((self.action_space,))]), verbose=0)
        X, Y = [], []
        for state, action, cur_q, reward, state_next, terminal in batch:
            if terminal:
                new_q = reward
            else:
                predicts = self.model.predict(state_next)
                max_expected_q = np.amax(predicts[0])
                new_q = (cur_q * (1 - Q_LEARN_RATE)) + (Q_LEARN_RATE * (reward + GAMMA * max_expected_q))
            y_q_values = self.model.predict(state)
            y_q_values[0][action] = new_q
            if STOCHASTIC_TRAIN:
                self.model.fit(state, y_q_values, steps_per_epoch=1, verbose=0)
            else:
                X.append(state[0])
                Y.append(y_q_values[0])

        if not STOCHASTIC_TRAIN:
            self.model.fit(np.array(X), np.array(Y), batch_size=len(X), verbose=0)

        if not TRAIN_EACH_STEP:
            self.memory = deque(maxlen=MEMORY_SIZE)

        self.isFit = True


def cartpole():
    env = gym.make(ENV_NAME)
    sl = score_logger.ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            # env.render()
            action, cur_q = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, cur_q, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                sl.add_score(step, run, dqn_solver.exploration_rate, len(dqn_solver.memory), refresh=run % PLOT_REFRESH == 0)
                break
            dqn_solver.experience_replay_new()


if __name__ == "__main__":
    cartpole()
