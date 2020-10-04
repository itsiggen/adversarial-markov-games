import numpy as np
import random
from replay_buffer import ReplayBuffer
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
import time

# List of hyper-parameters and constants
DECAY_RATE = 0 #0.80 #hoe hard de vorig geleerde waardes in het Q netwerk updaten 0.99 # pas aan in foolboxboundaryattack
BUFFER_SIZE = 100000
MINIBATCH_SIZE = 32
FINAL_EPSILON = 0.1
INITIAL_EPSILON = 1.0
TAU = 0.01 #om target model te laten trainen van echt model 0.01

class DeepQ(object):
    """Constructs the desired deep q learning network"""
    def __init__(self, input_shape,number_of_actions):
        self.input_shape = input_shape
        self.number_of_actions = number_of_actions
        self.construct_q_network()

    def construct_q_network(self):
        '''
        # Uses the network architecture found in DeepMind paper
        self.model = Sequential()
        self.model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, NUM_FRAMES)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(NUM_ACTIONS))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.00001))

        # Creates a target network as described in DeepMind paper
        self.target_model = Sequential()
        self.target_model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, NUM_FRAMES)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(64, 3, 3))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Flatten())
        self.target_model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.target_model.add(Dense(NUM_ACTIONS))
        self.target_model.compile(loss='mse', optimizer=Adam(lr=0.00001))
        self.target_model.set_weights(self.model.get_weights())
        '''
        
        self.model = Sequential()
        self.model.add(Dense(units = 64, activation='relu', input_shape=self.input_shape))
        self.model.add(Dense(units = 64, activation='relu'))
        self.model.add(Dense(units = self.number_of_actions))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.0001)) #0.00001
        
        self.target_model = Sequential()
        self.target_model.add(Dense(units = 64, activation='relu', input_shape=self.input_shape))
        self.target_model.add(Dense(units = 64, activation='relu'))
        self.target_model.add(Dense(units = self.number_of_actions))
        self.target_model.compile(loss='mse', optimizer=Adam(lr=0.0001)) #0.00001
        self.target_model.set_weights(self.model.get_weights())
        
        print("Successfully constructed networks.")

    # returnt de actie die je moet nemen
    def predict_action(self, data, epsilon):
        """Predict action of game controler where is epsilon
        probability randomly move."""
        q_actions = self.model.predict(data.reshape((1,)+self.input_shape), batch_size = 1)
        opt_policy = np.argmax(q_actions)
        rand_val = np.random.random()
        if rand_val < epsilon:
            opt_policy = np.random.randint(0, self.number_of_actions)
        return opt_policy #, q_actions[0, opt_policy]

    def train(self, s_batch, a_batch, r_batch, s2_batch, DECAY_RATE):
        """Trains network to fit given parameters"""
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, self.number_of_actions))

        for i in range(batch_size):
            targets[i] = self.model.predict(s_batch[i].reshape((1,)+self.input_shape), batch_size = 1)
            fut_action = self.target_model.predict(s2_batch[i].reshape((1,)+self.input_shape), batch_size = 1)
            targets[i, a_batch[i]] = r_batch[i]
            targets[i, a_batch[i]] += DECAY_RATE * np.max(fut_action)

        loss = self.model.train_on_batch(s_batch, targets)

    def save_network(self, path):
        # Saves model at specified path as h5 file
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        self.model = load_model(path)
        print("Succesfully loaded network.")

    def target_train(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)

