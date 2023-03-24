import pyspiel
import numpy as np
from open_spiel.python.bots import policy, uniform_random
from open_spiel.python.algorithms import evaluate_bots

# Load the saved policies
policy_path = "/path/to/saved/policies/"

#define players
player1 = uniform_random.UniformRandomBot(0,np.random)
player2 = uniform_random.UniformRandomBot(1,np.random)
#player_test = PolicyBot()
bots = [player1, player2]

#define game and start state
game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()

# Evaluate the policies on a game
results = np.array([evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random) for _ in range(1000)])
average_results = np.mean(results, axis=0)

print(average_results)

