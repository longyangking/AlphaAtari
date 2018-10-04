import numpy as np 
import argparse

__version__ = "0.0.1"
__author__ = "Yang Long"
__info__ = "Play Atari Game with AI"

__gym_game_names__ = {
    "AirRaid":"AirRaid-v0",
    #"Alien-ram":"Alien-ram-v0",
    "Alien":"Alien-v0",
    #"Amidar-ram":"Amidar-ram-v0",
    "Amidar":"Amidar-v0",
    "SpaceInvaders":"SpaceInvaders-v0"
}

if __name__=='__main__':

    parser = argparse.ArgumentParser(description=__info__)

    parser.add_argument("--retrain", action='store_true', default=False, help="Re-Train AI")
    parser.add_argument("--train",  action='store_true', default=False, help="Train AI")
    parser.add_argument("--verbose", action='store_true', default=False, help="Verbose")
    parser.add_argument("--playai", action='store_true', default=False, help="Play by AI")
    parser.add_argument("--play", action='store_true', default=False, help="Play by human")
    parser.add_argument("--name", default="SpaceInvaders", help='Game name')

    args = parser.parse_args()
    verbose = args.verbose
    game_name = args.name

    __filename__ = "model_{0}.h5".format(game_name)
    if game_name not in __gym_game_names__.keys():
        print("Error in game name: Not supported game yet. [{0}]".format(game_name))
        exit()

    gym_game_name = __gym_game_names__[game_name]

    if args.train:
        if verbose:
            print("Continue to train AI model for game: [{0}].".format(game_name))

        from ai import AI
        from train import TrainAI

        ai = AI(game_name=gym_game_name, verbose=verbose)
        if verbose:
            print("loading latest model: [{0}] ...".format(__filename__),end="")
        ai.load_nnet(__filename__)
        if verbose:
            print("load OK!")

        trainai = TrainAI(
            game_name=gym_game_name,
            ai=ai,
            verbose=verbose
        )
        trainai.start(filename=__filename__)

        if verbose:
            print("The latest AI model is saved as [{0}]".format(__filename__))

    if args.retrain:
        if verbose:
            print("Start to re-train AI model for game: [{0}].".format(game_name))

        from train import TrainAI

        trainai = TrainAI(game_name=gym_game_name, verbose=verbose)
        trainai.start(__filename__)

        if verbose:
            print("The latest AI model is saved as [{0}]".format(__filename__))

    if args.playai:
        if verbose:
            print("Start to play the game by the AI model, which will be rendered in the screen.")

        from ai import AI
        from atari import GameEngine

        ai = AI(game_name=gym_game_name, verbose=verbose)
        if verbose:
            print("loading latest model: [{0}] ...".format(__filename__),end="")
        ai.load_nnet(__filename__)
        if verbose:
            print("load OK!")

        print("Please close game in terminal after closing window (i.e, Press Ctrl+C).")
        engine = GameEngine(game_name=gym_game_name, verbose=verbose)
        engine.start_ai(ai=ai)

    if args.play:
        print("Play Atari game. Please close game in terminal after closing window (i.e, Press Ctrl+C).")

        from atari import GameEngine

        engine = GameEngine(game_name=gym_game_name, verbose=verbose)
        engine.start()