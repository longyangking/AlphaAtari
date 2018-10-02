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
    parser.add_argument("--playbyai", action='store_true', default=False, help="Play by AI")
    parser.add_argument("--play", action='store_true', default=False, help="Play by human")
    parser.add_argument("--name", default="AirRaid", help='Game name')

    args = parser.parse_args()
    verbose = args.verbose
    game_name = args.name

    __filename__ = "model_{0}.h5".format(game_name)
    if game_name not in __gym_game_names__.keys():
        print("Error in game name: Not supported game yet. [{0}]".format(game_name))
        exit()

    if args.train:
        if verbose:
            print("Continue to train AI model for game: [{0}].".format(game_name))


        if verbose:
            print("The latest AI model is saved as [{0}]".format(__filename__))

    if args.retrain:
        if verbose:
            print("Start to re-train AI model for game: [{0}].".format(game_name))


        if verbose:
            print("The latest AI model is saved as [{0}]".format(__filename__))

    if args.playbyai:
        if verbose:
            print("Start to play the game by the AI model, which will be rendered in the screen.")

    if args.play:
        print("Play BulletScreen game. Please close game in terminal after closing window (i.e, Press Ctrl+C).")