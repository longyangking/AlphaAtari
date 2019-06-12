from __future__ import print_function

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, Add, Concatenate
from keras.optimizers import SGD, Adam
from keras import regularizers
import keras.backend as K
import tensorflow as tf

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only error will be shown

class NeuralNetwork:
    def __init__(self, input_shape, output_dim, network_structure,
        feature_dim=32, learning_rate=1e-3, l2_const=1e-4, verbose=False
    ):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.network_structure = network_structure
        self.feature_dim = feature_dim

        self.learning_rate = learning_rate
        self.l2_const = l2_const
        self.verbose = verbose

        self.model = self.build_model()

    def build_model(self):
        state_tensor = Input(shape=self.input_shape)

        x = self.__conv_block(state_tensor, self.network_structure[0]['filters'], self.network_structure[0]['kernel_size'])
        if len(self.network_structure) > 1:
            for h in self.network_structure[1:]:
                x = self.__res_block(x, h['filters'], h['kernel_size'])

        action_prob_tensor = self.__action_prob_block(x)
        model = Model(inputs=state_tensor, outputs=action_prob_tensor)
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

    def __dense_block(self, x, units):
        out = Dense(
            units,
            use_bias=False,
            activation='linear',
            kernel_regularizer= regularizers.l2(self.l2_const)
		)(x)
        out = LeakyReLU()(out)
        return out

    def __action_prob_block(self, x):
        out = self.__conv_block(x, 64, kernel_size=3)
        out = Flatten()(out)
        out = self.__dense_block(x, units=32)

        action_prob = Dense(
			self.output_dim, 
            use_bias=False,
            activation='sigmoid',
            kernel_regularizer=regularizers.l2(self.l2_const),
			)(out)

        return action_prob

    def __icm_forward_block(self, action_prob_tensor, phi_t_tensor):
        x = self.__dense_block(action_prob_tensor, units=32)
        y = self.__dense_block(phi_t_tensor, units=64)
        out = Concatenate()([x, y])

        out = self.__dense_block(out, units=64)
        out = self.__dense_block(out, units=64)
        out = self.__dense_block(out, units=64)
        out = self.__dense_block(out, units=64)
        return out

    def __icm_features_block(self, state_shape):
        out = self.__conv_block(self, state_tensor, filters=64, kernel_size=3)
        out = self.__conv_block(self, out, filters=32, kernel_size=3)
        out = self.__conv_block(self, out, filters=32, kernel_size=3)
        out = Flatten()(out)
        out = self.__dense_block(out, units=32)
        out = Dense(
            units=self.feature_dim,
            use_bias=False,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_const)
        )(out)
        return out

    def __icm_inverse_block(self, phi1, phi2):
        phi1 = self.__dense_block(phi1, units=32)
        phi2 = self.__dense_block(phi2, units=32)
        phi = Concatenate()([phi1, phi2])
        out = self.__dense_block(phi, units=64)
        out = self.__dense_block(out, units=32)
        action_prob = Dense(
            self.output_dim,
            use_bias=False,
            activation='sigmoid',
            kernel_regularizer=regularizers.l2(self.l2_const)
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

        # The network structure is based on intrinsic curiosity module (ICM)

        network_structure = dict()
        network_structure['policy'] = list()
        network_structure['policy'].append({'filters':64, 'kernel_size':3})
        network_structure['policy'].append({'filters':64, 'kernel_size':3})
        network_structure['policy'].append({'filters':64, 'kernel_size':3})
        network_structure['policy'].append({'filters':64, 'kernel_size':3})

        network_structure['icm'] = dict()
        network_structure['icm']['forward_model'] = list()
        network_structure['icm']['features'] = list()
        network_structure['icm']['inverse_model'] = list()

        network_structure['icm']['forward_model'].append({'filters':64, 'kernel_size':3})
        network_structure['icm']['forward_model'].append({'filters':64, 'kernel_size':3})
        network_structure['icm']['forward_model'].append({'filters':64, 'kernel_size':3})
        network_structure['icm']['forward_model'].append({'filters':64, 'kernel_size':3})

        network_structure['icm']['features'].append({'filters':64, 'kernel_size':3})
        network_structure['icm']['features'].append({'filters':64, 'kernel_size':3})
        network_structure['icm']['features'].append({'filters':64, 'kernel_size':3})
        network_structure['icm']['features'].append({'filters':64, 'kernel_size':3})

        network_structure['icm']['inverse_model'].append({'filters':64, 'kernel_size':3})
        network_structure['icm']['inverse_model'].append({'filters':64, 'kernel_size':3})
        network_structure['icm']['inverse_model'].append({'filters':64, 'kernel_size':3})
        network_structure['icm']['inverse_model'].append({'filters':64, 'kernel_size':3})

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