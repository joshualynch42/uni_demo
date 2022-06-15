import pandas as pd
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Concatenate, Input
from keras.optimizers import adam_v2
from collections import deque
import random, os
import tensorflow as tf

class Replay_memory():
    def __init__(self, max_replay_size, episodes, beta=0.4, priority_scale=0.6):
        self.replay_memory = deque(maxlen=max_replay_size)
        self.replay_priorities = deque(maxlen=max_replay_size)
        self.priority_scale = priority_scale
        self.beta = beta
        self.delta_beta = (1 - beta) / episodes

    def add(self, transition):
        self.replay_memory.append(transition)
        self.replay_priorities.append(max(self.replay_priorities, default=1))

    def update_hype(self):
        self.beta += self.delta_beta

    def get_probabilities(self, priority_scale): # priority scale is alpha
        scaled_priorities = np.array(self.replay_priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities) # pi^a / sum(pi)
        return sample_probabilities

    def get_importance(self, probabilities, beta):
        importance = (1/len(self.replay_memory)) * (1/probabilities)
        importance_normalised = importance / max(importance)
        return importance_normalised

    def sample(self, batch_size):
        sample_probs = self.get_probabilities(self.priority_scale)
        sample_indicies = random.choices(range(len(self.replay_memory)), k=batch_size, weights=sample_probs)
        samples = np.array(self.replay_memory)[sample_indicies]
        importance = self.get_importance(sample_probs[sample_indicies], self.beta)
        return samples, importance, sample_indicies

    def update_priorities(self, indicies, td_errors, offset=0.1):
        for i, td in zip(indicies, td_errors):
            self.replay_priorities[i] = abs(td) + offset


class transfer_dueling_ddqn_per():
    def __init__(self, env, rl_params):
        self.label = 'Dueling Double Per'
        self.replay_memory_size = rl_params['replay_memory_size']
        self.experience_replay = Replay_memory(self.replay_memory_size, rl_params['episodes'])
        self.discount = rl_params['discount']
        self.min_replay_memory_size = rl_params['min_replay_memory_size']
        self.minibatch_size = rl_params['minibatch_size']
        self.epsilon_decay = rl_params['epsilon_decay']
        self.min_epsilon = rl_params['min_epsilon']
        self.action_space = env.action_space_size
        self.epsilon = rl_params['epsilon']
        self.model = self.create_model(env, env.state['vec']) # main model trained every step
        self.target_model = self.create_model(env, env.state['vec'])
        self.target_model.set_weights(self.model.get_weights())
        self.sim_model = self.create_model(env, env.state['vec']) # knowledge from sim
        self.target_update_counter = 0
        self.update_target_every = rl_params['update_target_every'] # every 5 episodes

    def create_model(self, env, goal):

        input_img = Input(shape=env.img_shape)
        x = Sequential()(input_img)
        x = Conv2D(32, 8, activation='relu', strides=(4,4))(x)
        x = Conv2D(64, 4, activation='relu', strides=(2,2))(x)
        x = Conv2D(64, 3, activation='relu', strides=(1,1))(x)
        x = Flatten()(x)
        # second network
        input_vec = Input(shape=(len(goal)))
        y = Dense(8, activation='relu')(input_vec)  # change one-hot to layer
        # merged network
        z = Concatenate()([x, y])
        z = Dense(512, activation="relu")(z) #64

        # advantage stream
        A = Dense(self.action_space, activation="linear")(z)
        # value stream
        V = Dense(1, activation="linear")(z)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        model = Model(inputs=(input_img, input_vec), outputs=Q)
        model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=0.0001), metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        # current_state, action, reward, new_state, done
        # self.replay_memory.append(transition)
        self.experience_replay.add(transition)

    def get_qs(self, state):
        img = state['img']
        vec = state['vec']
        return self.model.predict((np.array(img).reshape(-1, *img.shape),
                                np.array(vec).reshape(-1, *vec.shape)))[0]

    def get_qs_know(self, state):
        img = state['img']
        vec = state['vec']
        return self.sim_model.predict((np.array(img).reshape(-1, *img.shape),
                                np.array(vec).reshape(-1, *vec.shape)))[0]

    def act(self, current_state):
        if np.random.random() > self.epsilon:
            dest_qs = self.get_qs(current_state)
            know_qs = self.get_qs_know(current_state)
            action = np.argmax(dest_qs+know_qs) # adds knowledge and current qs together q
        else:
            action = np.random.randint(0, self.action_space)

        # Decay Epsilon
        if len(self.experience_replay.replay_memory) > self.min_replay_memory_size:
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.min_epsilon, self.epsilon)
        return action

    def train(self, terminal_state):
        if len(self.experience_replay.replay_memory) < self.min_replay_memory_size:
            return

        # minibatch = random.sample(self.replay_memory, self.minibatch_size)
        minibatch, importance, sample_indicies = self.experience_replay.sample(self.minibatch_size)

        ## current states = [dict{img, vec}, dict{img, vec}, dict{img, vec}......]
        ## seperating dicts into arrays
        current_states = np.array([transition[0] for transition in minibatch])  # Add divide by max (scale results)
        img_arr = np.array([x['img'] for x in current_states])
        vec_arr = np.array([x['vec'] for x in current_states])
        current_qs_list = self.model.predict((img_arr, vec_arr))

        new_current_states = np.array([transition[3] for transition in minibatch])  # Add divide by max (scale results)
        fut_img_arr = np.array([x['img'] for x in new_current_states])
        fut_vec_arr = np.array([x['vec'] for x in new_current_states])
        future_qs_list = self.target_model.predict((fut_img_arr , fut_vec_arr))

        x_img = []  # array of current states
        x_vec = []
        y = []  # array of new q values
        td_errors = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            old_q = current_qs_list[index][action]
            td_error = new_q - old_q # maybe use target network for old q
            td_errors.append(td_error)

            current_qs = current_qs_list[index]
            current_qs[action] = old_q + (importance[index] * td_error)
            x_img.append(current_state['img'])
            x_vec.append(current_state['vec'])
            y.append(current_qs)

        self.experience_replay.update_priorities(sample_indicies, td_errors)
        self.model.fit((np.array(x_img), np.array(x_vec)), np.array(y), batch_size=self.minibatch_size, verbose=0, shuffle=False)

        if terminal_state:
            self.target_update_counter += 1
        # updating target models
        if self.target_update_counter > self.update_target_every:
            self.target_update_counter = 0
            self.target_model.set_weights(self.model.get_weights())

    def render(self, reward_arr):
        pass

    def save_model(self, model_dir):
        self.model.save(model_dir)

        print('Agent saved as '.format(model_dir))

    def load_model(self, model_dir):
        self.sim_model = keras.models.load_model(model_dir)
        print('Agent {} has loaded'.format(model_dir))
