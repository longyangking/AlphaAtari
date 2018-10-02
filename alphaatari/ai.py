from __future__ import print_function

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, Add
from keras.optimizers import SGD, Adam
from keras import regularizers
import keras.backend as K
import tensorflow as tf

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only error will be shown

class NeuralNetwork:
    def __init__(self, input_shape, output_dim, network_structure,
        learning_rate=1e-3, l2_const=1e-4, verbose=False
    ):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.network_structure = network_structure

        self.learning_rate = learning_rate
        self.l2_const = l2_const
        self.verbose = verbose

        self.model = self.build_model()

    def build_model(self):

    def __conv_block(self, x, filters, kernel_size):

    def __res_block(self, x, filters, kernel_size):

    def __action_block(self, x):

    def __value_block(self, x):

    def fit(self, states, actions, values, epochs, batch_size):
        history = self.model.fit(states, [actions, values], epochs=epochs, batch_size=batch_size)
        return history

    def update(self, states, actions, values):
        loss = self.model.update(states, [actions, values])
        return loss

    def predict(self, state):
        state = np.array(state)
        states = state.reshape(1, self.input_shape)
        actions, values = self.model.predict(states)
        return actions[0], values[0]

    def save_model(self, filename):
        self.model.save_weights(filename)

    def load_model(self, filename):
        self.model.load_weights(filename)

    def plot_model(self, filename):
        from keras.utils import plot_model
        plot_model(self.model, show_shapes=True, to_file=filename)

class AI:
    def __init__(self, game_name, verbose=False):
        self.game_name = game_name
        self.verbose = verbose

        from atari import AtariGame
        self.state_shape = AtariGame(game_name=self.game_name).get_observation_shape()
        self.action_dim = AtariGame(game_name=self.game_name).get_action_dimension()

        network_structure = list()
        network_structure.append({'filters':64, 'kernel_size':3})
        network_structure.append({'filters':64, 'kernel_size':3})
        network_structure.append({'filters':64, 'kernel_size':3})
        network_structure.append({'filters':64, 'kernel_size':3})

        self.nnet = NeuralNetwork(
            input_shape=self.state_shape,
            output_dim=self.action_dim,
            network_structure=network_structure,
            verbose=self.verbose
        )

    def get_state_shape(self):
        return np.copy(self.state_shape)

    def get_action_dimension(self):
        return self.action_dim

    def train(self, dataset, epochs, batch_size):
        states, actions, values = dataset
        history = self.nnet.fit(states, actions, values, epochs=epochs, batch_size=batch_size)
        return history

    def update(self, dataset):
        states, actions, values = dataset
        loss = self.nnet.update(states, actions, values)
        return loss

    def evaluate_function(self, state):
        action_prob, value = self.nnet.predict(state)
        return action_prob, value

    def play(self, state):
        action_prob, value = self.evaluate_function(state)
        action = np.argmax(action_prob)
        return action

    def save_nnet(self, filename):
        self.nnet.save_model(filename)

    def load_nnet(self, filename):
        self.nnet.load_model(filename)

    def plot_nnet(self, filename):
        self.nnet.plot_model(filename)