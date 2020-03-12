# -*- coding: utf-8 -*-
import os
import gym
import random
import numpy as np

from collections import deque

import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K


class DQN:
    def __init__(self):
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        if os.path.exists('dqn.h5'):
            self.model.load_weights('dqn.h5')

        self.memory_buffer = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.env = gym.make('CartPole-v1')

    def build_model(self):
        inputs = Input(shape=(4,))
        x = Dense(16, activation='relu')(inputs)
        x = Dense(16, activation='relu')(x)
        x = Dense(2, activation='linear')(x)

        model = Model(inputs=inputs, outputs=x)

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def egreedy_action(self, state):
        if np.random.rand() <= self.epsilon:
             return random.randint(0, 1)
        else:
            q_values = self.model.predict(state)[0]
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory_buffer.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def process_batch(self, batch_size):
        data_sample = random.sample(self.memory_buffer, batch_size)
        # tate, action, reward, next_state, done
        states = np.array([data[0] for data in data_sample])
        next_states = np.array([data[3] for data in data_sample])

        y = self.model.predict(states)
        q_vals = self.target_model.predict(next_states)

        for i, (_, action, reward, _, done) in enumerate(data_sample):
            target = reward
            if not done:
                target += self.gamma * np.amax(q_vals[i])
            y[i][action] = target

        return states, y


    def train(self, episode, batch):
        self.model.compile(loss='mse', optimizer=Adam())

        train_reward = []
        train_loss = []
        count = 0
        for i in range(episode):
            observation = self.env.reset()
            reward_sum = 0
            loss = np.infty
            done = False

            while not done:
                x = observation.reshape(-1, 4)
                action = self.egreedy_action(x)
                observation, reward, done, _ = self.env.step(action)
                reward_sum += reward
                self.remember(x[0], action, reward, observation, done)

                if len(self.memory_buffer) > batch:
                    X, y = self.process_batch(batch)
                    loss = self.model.train_on_batch(X, y)

                    count += 1
                    self.update_epsilon()

                    if count != 0 and count % 20 == 0:
                        self.update_target_model()
            if i%5==0:
                train_reward.append(reward_sum)
                train_loss.append(loss)
            print('%%%.2f'%(100*i/episode))
    
        self.model.save_weights('dqn.h5')
        epochs = range(1, len(train_reward)+1)
        plt.plot(epochs,train_reward,'b', label='Training reward')
        plt.legend()
        plt.show()
        plt.clf()
        plt.plot(epochs,train_loss,'bo', label='Training loss')
        plt.legend()
        plt.show()

    def play(self):
        observation = self.env.reset()

        count = 0
        reward_sum = 0
        random_episodes = 0

        while random_episodes < 8:
            self.env.render()

            x = observation.reshape(-1, 4)
            q_values = self.model.predict(x)[0]
            action = np.argmax(q_values)
            observation, reward, done, _ = self.env.step(action)

            count += 1
            reward_sum += reward

            if done:
                print("Reward for this episode was: {}, turns was: {}".format(reward_sum, count))
                random_episodes += 1
                reward_sum = 0
                count = 0
                observation = self.env.reset()

        self.env.close()


model = DQN()
model.train(300, 32)
model.play()
