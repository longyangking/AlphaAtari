import numpy as np 
from ai import AI

class SelfplayEngine:
    def __init__(self, game_name, ai, verbose=False):
        self.game_name = game_name
        self.ai = ai
        self.verbose = verbose

        self.states = list()
        self.actions = list()
        self.scores = list()
        self.obervations = list()

    def get_state(self):
        '''
        Get the latest state
        '''
        return self.states[-1]

    def update_states(self):
        '''
        Update the list of states
        '''
        observation = self.obervations[-1]  # Currently, the state is equal to the observation
        self.states.append()

    def start(self):
        '''
        The main process for self-play engine
        '''
        
        # TODO The main process for self-play engine

        states = np.array(self.states)
        action_probs = list()
        values = list()

        return states, action_probs, values

class TrainAI:
    def __init__(self, game_name, ai=None, verbose=False):
        self.game_name = game_name
        self.verbose = verbose
        
        if ai is None:
            self.ai = AI(game_name=self.game_name, verbose=self.verbose)
        else:
            self.ai = ai    

    def get_selfplay_data(self, n_rounds):
        states = list()
        action_probs = list()
        values = list()

        for i in range(n_rounds):
            engine = SelfplayEngine(game_name=game_name, ai=self.ai, verbose=self.verbose)
            _states, _action_probs, _values = engine.start()

            # TODO merge these self-play data
        
        return states, action_probs, values

    def update_ai(self, dataset):
        '''
        Updation of AI mode
        '''
        if self.verbose:
            print("Updating the neural network of AI model...")

        history = self.ai.train(dataset, epochs=30, batch_size=32)
        loss = history.history['loss']

        if self.verbose:
            print("End of updation with loss [{0}].".format(loss))
            
        return loss

    def start(self, filename):
        '''
        Main training process
        '''
        n_epochs = 1000
        n_rounds = 30
        n_checkpoints = 10

        if self.verbose:
            print("Train AI model with epochs [{0}]".format(n_epochs))

        for i in range(n_epochs):
            if self.verbose:
                print("{0}th self-play training process ...".format(i+1))

            dataset = self.get_selfplay_data(n_rounds)

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