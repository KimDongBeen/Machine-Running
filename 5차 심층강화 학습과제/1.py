import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

EPOCHS = 50
THRESHOLD = 15
MONITOR = False


class DQN():
    def __init__(self, env_string, batch_size=64, IM_SIZE=84, m=4, target_update=15,
                 logfile_name='./runs/NAME'): 
        self.memory = deque(maxlen=5000)
        self.env = gym.make(env_string)
        input_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = 1.0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.IM_SIZE = IM_SIZE
        self.m = m
        self.writer = tf.summary.create_file_writer(logfile_name)

        alpha = 0.00025
        # alpha_decay=0.01
        if MONITOR: self.env = gym.wrappers.Monitor(self.env, '../data/' + env_string, force=True)

        # Init model
        self.model = Sequential()
        self.model.add(Conv2D(32, 8, (4, 4), activation='relu', padding='valid', input_shape=(IM_SIZE, IM_SIZE, m)))
        # self.model.add(MaxPooling2D())
        self.model.add(Conv2D(64, 4, (2, 2), activation='relu', padding='valid'))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(64, 3, (1, 1), activation='relu', padding='valid'))
        self.model.add(MaxPooling2D())
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='elu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=alpha, clipnorm=1.0))
        self.model_target = tf.keras.models.clone_model(self.model)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()

        else:
            return np.argmax(self.model.predict(state))

    def preprocess_state(self, img):
        img_temp = img[31:195]  # Choose the important area of the image
        img_temp = tf.image.rgb_to_grayscale(img_temp)
        img_temp = tf.image.resize(img_temp, [self.IM_SIZE, self.IM_SIZE],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img_temp = tf.cast(img_temp, tf.float32)

        return img_temp[:, :, 0]

    def combine_images(self, img1, img2):
        if len(img1.shape) == 3 and img1.shape[0] == self.m:
            im = np.append(img1[1:, :, :], np.expand_dims(img2, 0), axis=2)
            return tf.expand_dims(im, 0)
        else:
            im = np.stack([img1] * self.m, axis=2)
            return tf.expand_dims(im, 0)
        # return np.reshape(state, [1, 4])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model_target.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

    def train(self):
        scores = deque(maxlen=100)
        avg_scores = []

        for e in range(EPOCHS):
            state = self.env.reset()
            state = self.preprocess_state(state)
            state = self.combine_images(state, state)
            done = False
            i = 0
            while not done:
                action = self.choose_action(state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                next_state = self.combine_images(next_state, state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)  # decrease epsilon
                i += reward

            scores.append(i)
            mean_score = np.mean(scores)
            avg_scores.append(mean_score)
            with self.writer.as_default():
                tf.summary.scalar("avg_score", mean_score, step=e)
                self.writer.flush()
            if mean_score >= THRESHOLD:
                print('Solved after {} trials ✔'.format(e))
                break  # return avg_scores
            if e % 10 == 0:
                print('[Episode {}] - Average Score: {}.'.format(e, mean_score))

            if e % self.target_update == 0:
                self.model_target.set_weights(self.model.get_weights())

            self.replay(self.batch_size)

        print('Did not solve after {} episodes 😞'.format(e))

   
        self.save_model()

        return avg_scores

    def save_model(self, model_path='saved_model/model.pb'):
      
        print('\nsave model : \"{}\"'.format(model_path))
        self.model.save(model_path)
        pass

    def load_model(self, model_path='saved_model/model.pb'):
     
        print('\nload model : \"{}\"'.format(model_path))
        self.model = tf.keras.models.load_model(model_path)
        pass

    def test(self):
     
        scores1 = deque(maxlen=100)
        avg_scores = []
        print('\ntest model')

        for e in range(1):
            state = self.env.reset()
            state = self.preprocess_state(state)
            state = self.combine_images(state, state)

            done = False
            i = 0

            while not done:
                self.env.render()
                action = self.choose_action(state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                next_state = self.combine_images(next_state, state)
                self.remember(state, action, reward, next_state, done)
                state = next_state

                i += reward

            scores1.append(i)
            mean_score = np.mean(scores1)
            avg_scores.append(mean_score)
            with self.writer.as_default():
                tf.summary.scalar("avg_score", mean_score, step=e)
                self.writer.flush()
            if mean_score >= THRESHOLD:
                print('Solved after {} trials ✔'.format(e))
                break  # return avg_scores

            self.replay(self.batch_size)

        return avg_scores
        pass


if __name__ == '__main__':
    tf.keras.backend.clear_session()
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)
    env_string = 'BreakoutDeterministic-v4'
    agent = DQN(env_string)
    print("Main Model", agent.model.summary())
    print("Target Model", agent.model_target.summary())
    scores = agent.train()
    plt.plot(scores)
    plt.show()
    scores1 = agent.test()
    print(scores1)
    AVG = np.mean(scores1)
    print(AVG)
    agent.env.close()
