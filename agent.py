import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import random
from collections import deque


class BitcoinTradingAgent:
    def __init__(self, state_size, action_size, max_investment=0.5, learning_rate=0.001, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.max_investment = max_investment
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(32, input_shape=(self.state_size), return_sequences=True))
        model.add(LSTM(16))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            confidence = random.uniform(0, 1)
        else:
            act_values = self.model.predict(state, verbose=0)
            action = np.argmax(act_values[0])
            confidence = act_values[0][action]
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return action, confidence

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
