import pyspiel
import numpy as np
from open_spiel.python.bots import uniform_random
from open_spiel.python.algorithms import evaluate_bots
from open_spiel.python import rl_environment
from open_spiel.python.bots.policy import PolicyBot
from open_spiel.python import policy


# Load the saved policies
policy_path = "/path/to/saved/policies/"
game = pyspiel.load_game_as_turn_based("tic_tac_toe")

random_policy = policy.UniformRandomPolicy(game)
player1 = PolicyBot(0, np.random.RandomState(4321), random_policy)
player2 = uniform_random.UniformRandomBot(1, np.random.RandomState(4321))

# Evaluate the policies on a game
env = rl_environment.Environment(game) #This module wraps Open Spiel Python interface providing an RL-friendly API
results = np.array([evaluate_bots.evaluate_bots(env.game.new_initial_state(), [player1,player2], np.random.RandomState(4321)) for _ in range(10)])

