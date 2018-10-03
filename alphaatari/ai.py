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
        input_tensor = Input(shape=self.input_shape)

        x = self.__conv_block(input_tensor, self.network_structure[0]['filters'], self.network_structure[0]['kernel_size'])
        if len(self.network_structure) > 1:
            for h in self.network_structure[1:]:
                x = self.__res_block(x, h['filters'], h['kernel_size'])

        action_prob_tensor = self.__action_prob_block(x)
        model = Model(inputs=input_tensor, outputs=action_prob_tensor)
        model.compile(
            loss='mse',
            optimizer=Adam(self.learning_rate)
        )

        return model

    def __conv_block(self, x, filters, kernel_size):
        out = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            padding = 'same',
            activation='linear',
            kernel_regularizer = regularizers.l2(self.l2_const)
        )(x)
        out = BatchNormalization(axis=1)(out)
        out = LeakyReLU()(out)
        return out

    def __res_block(self, x, filters, kernel_size):
        out = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            padding = 'same',
            activation='linear',
            kernel_regularizer = regularizers.l2(self.l2_const)
        )(x)
        out = BatchNormalization(axis=1)(out)
        out = Add()([out, x])
        out = LeakyReLU()(out)
        return out

    def __action_prob_block(self, x):
        out = Conv2D(
            filters = 64,
            kernel_size = (3,3),
            padding = 'same',
            activation='linear',
            kernel_regularizer = regularizers.l2(self.l2_const)
        )(x)
        out = BatchNormalization(axis=1)(out)
        out = LeakyReLU()(out)

        out = Flatten()(out)
        out = Dense(
            36,
            use_bias=False,
            activation='linear',
            kernel_regularizer= regularizers.l2(self.l2_const)
		)(out)
        out = LeakyReLU()(out)

        action_prob = Dense(
			self.output_dim, 
            use_bias=False,
            activation='sigmoid',
            kernel_regularizer=regularizers.l2(self.l2_const),
			)(out)

        return action_prob

    def fit(self, states, action_probs, epochs, batch_size):
        history = self.model.fit(states, action_probs, epochs=epochs, batch_size=batch_size)
        return history

    def update(self, states, action_probs, values):
        loss = self.model.update(states, action_probs)
        return loss

    def predict(self, state):
        state = np.array(state)
        states = state.reshape(1, *self.input_shape)
        action_probs = self.model.predict(states)
        return action_probs[0]

    def save_model(self, filename):
        self.model.save_weights(filename)

    def load_model(self, filename):
        self.model.load_weights(filename)

    def plot_model(self, filename):
        from keras.utils import plot_model
        plot_model(self.model, show_shapes=True, to_file=filename)

class AI:
    def __init__(self, game_name, time_span=3, verbose=False):
        self.game_name = game_name
        self.verbose = verbose

        from atari import AtariGame
        observation_shape = AtariGame(game_name=self.game_name).get_observation_shape()
        self.state_shape = *observation_shape, time_span
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
        states, action_probs = dataset
        history = self.nnet.fit(states, action_probs, epochs=epochs, batch_size=batch_size)
        return history

    def update(self, dataset):
        states, action_probs = dataset
        loss = self.nnet.update(states, action_probs)
        return loss

    def evaluate_function(self, state):
        action_prob = self.nnet.predict(state)
        eps = 1e-12
        action_prob = action_prob/np.sum(action_prob + eps)
        return action_prob

    def play(self, state):
        action_prob = self.evaluate_function(state)
        action = np.argmax(action_prob)
        return action

    def save_nnet(self, filename):
        self.nnet.save_model(filename)

    def load_nnet(self, filename):
        self.nnet.load_model(filename)

    def plot_nnet(self, filename):
        self.nnet.plot_model(filename)