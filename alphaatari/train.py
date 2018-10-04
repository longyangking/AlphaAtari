import numpy as np 
from ai import AI 
import time
from atari import AtariGame

class SelfplayEngine:
    def __init__(self, game_name, ai, verbose=False):
        self.game_name = game_name
        self.ai = ai
        self.verbose = verbose
        self.state_shape = ai.get_state_shape()

        self.states = list()
        self.actions = list()
        self.rewards = list()
        self.observations = list()

    def get_state(self):
        '''
        Get the latest state
        '''
        return self.states[-1]

    def update_states(self):
        '''
        Update the list of states
        '''
        Nx, Ny, channel = self.state_shape
        state = np.zeros((Nx,Ny,channel))
        n_observations = len(self.observations)
        for i in range(channel):
            if i+1 <= n_observations:
                state[:,:,-(i+1)] = self.observations[-(i+1)]

        self.states.append(state)

    def start(self, epsilon, gamma, timesteps=100):
        '''
        The main process for self-play engine
        '''
        
        game = AtariGame(game_name=self.game_name, verbose=self.verbose)
        self.observations.append(game.get_observation())
        self.update_states()

        done = False
        t = 0
        while (not done) and (t < timesteps):
            #game.render()
            state = self.get_state()

            v = np.random.random()
            if v > epsilon:
                action = game.get_random_action()
            else:
                action = self.ai.play(state)
            
            _, reward, done, info = game.step(action)

            self.actions.append(action)
            self.observations.append(game.get_observation())
            self.update_states()
            self.rewards.append(reward)
            t += 1

        if self.verbose and (t >= timesteps):
            print("End of self-play game due to the limit of time-steps [{0}].".format(timesteps))

        # The one of the core aspects in deep learing is to define an appropriate merit function

        eps = 1e-12
        action_probs = list()
        N = len(self.rewards)
        for i in range(N):
            action_prob = self.ai.evaluate_function(self.states[N-1-i])
            if i == 0:
                action_prob[self.actions[N-1-i]] = 0
            if i == N-1:
                action_prob[self.actions[N-1-i]] = self.rewards[N-1-i]
            else:
                action_prob[self.actions[N-1-i]] += self.rewards[N-1-i] - self.rewards[N-2-i]
            
            action_prob = action_prob/np.sum(action_prob + eps)
            action_probs.append(action_prob)

        states = self.states[1:]

        return states, action_probs

class TrainAI:
    def __init__(self, game_name, ai=None, time_span=3, verbose=False):
        self.game_name = game_name
        self.verbose = verbose
        self.time_span = time_span
        
        if ai is None:
            self.ai = AI(game_name=self.game_name, time_span=time_span, verbose=self.verbose)
        else:
            self.ai = ai    

        self.losses = list()

    def get_losses(self):
        return self.losses

    def get_selfplay_data(self, n_rounds, epsilon=0.5, gamma=0.9, timesteps=100):
        states = list()
        action_probs = list()
        values = list()

        if self.verbose:
            starttime = time.time()

        for i in range(n_rounds):
            if self.verbose:
                print("{0}th round of self-play process ...".format(i+1))

            engine = SelfplayEngine(game_name=self.game_name, ai=self.ai, verbose=self.verbose)
            _states, _action_probs = engine.start(epsilon=epsilon, gamma=gamma, timesteps=timesteps)

            for n in range(len(_states)):
                states.append(_states[n])
                action_probs.append(_action_probs[n])

        if self.verbose:
            endtime = time.time()
            print("End of self-play process with data size [{0}] and cost time [{1:.1f}s].".format(len(states),(endtime - starttime)))

        states = np.array(states)
        action_probs = np.array(action_probs)
        
        return states, action_probs

    def update_ai(self, dataset):
        '''
        Updation of AI mode
        '''
        if self.verbose:
            print("Updating the neural network of AI model...")

        history = self.ai.train(dataset, epochs=1, batch_size=32)
        loss = history.history['loss']

        if self.verbose:
            print("End of updation with loss [{0}].".format(loss))
            
        return loss

    def start(self, filename):
        '''
        Main training process
        '''
        n_epochs = 1000
        n_rounds = 1
        n_checkpoints = 1
        timesteps = 100

        if self.verbose:
            print("Train AI model with epochs [{0}]".format(n_epochs))

        for i in range(n_epochs):
            if self.verbose:
                print("{0}th self-play training process with rounds [{1}]".format(i+1, n_rounds))

            dataset = self.get_selfplay_data(n_rounds, timesteps=timesteps)

            loss = self.update_ai(dataset)
            self.losses.extend(loss)

            if self.verbose:
                print("End of training process.")

            if (i+1)%n_checkpoints == 0:
                if self.verbose:
                    print("Checkpoint: Saving AI model with filename [{0}] ...".format(filename),end="")

                self.ai.save_nnet(filename)

                if self.verbose:
                    print("OK!")