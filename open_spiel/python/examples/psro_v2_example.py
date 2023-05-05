# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example running PSRO on OpenSpiel Sequential games.

To reproduce results from (Muller et al., "A Generalized Training Approach for
Multiagent Learning", ICLR 2020; https://arxiv.org/abs/1909.12823), run this
script with:
  - `game_name` in ['kuhn_poker', 'leduc_poker']
  - `n_players` in [2, 3, 4, 5]
  - `meta_strategy_method` in ['alpharank', 'uniform', 'nash', 'prd']
  - `rectifier` in ['', 'rectified']

The other parameters keeping their default values.
"""

import time

from absl import app
from absl import flags
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt

# pylint: disable=g-bad-import-order
import pyspiel
import tensorflow.compat.v1 as tf
# pylint: enable=g-bad-import-order

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python.algorithms.psro_v2 import best_response_oracle
from open_spiel.python.algorithms.psro_v2 import psro_v2
from open_spiel.python.algorithms.psro_v2 import rl_oracle
from open_spiel.python.algorithms.psro_v2 import rl_policy
from open_spiel.python.algorithms.psro_v2 import strategy_selectors
from open_spiel.python.algorithms.psro_v2.utils import save_object, load_object

from open_spiel.python.bots.policy import PolicyBot
from matplotlib import colors
from open_spiel.python.algorithms import evaluate_bots


FLAGS = flags.FLAGS #flag, default, description

# Game-related
flags.DEFINE_string("game_name", "kuhn_poker", "Game name.")  
flags.DEFINE_integer("n_players", 2, "The number of players.")

# PSRO related
flags.DEFINE_string("meta_strategy_method", "alpharank",
                    "Name of meta strategy computation method.")
flags.DEFINE_integer("number_policies_selected", 1,
                     "Number of new strategies trained at each PSRO iteration.")
flags.DEFINE_integer("sims_per_entry", 1000,
                     ("Number of simulations to run to estimate each element"
                      "of the game outcome matrix."))

flags.DEFINE_integer("gpsro_iterations", 100,
                     "Number of training steps for GPSRO.")
flags.DEFINE_bool("symmetric_game", False, "Whether to consider the current "
                  "game as a symmetric game.")

# Rectify options
flags.DEFINE_string("rectifier", "",
                    "Which rectifier to use. Choices are '' "
                    "(No filtering), 'rectified' for rectified.")
flags.DEFINE_string("training_strategy_selector", "probabilistic",
                    "Which strategy selector to use. Choices are "
                    " - 'top_k_probabilities': select top "
                    "`number_policies_selected` strategies. "
                    " - 'probabilistic': Randomly samples "
                    "`number_policies_selected` strategies with probability "
                    "equal to their selection probabilities. "
                    " - 'uniform': Uniformly sample `number_policies_selected` "
                    "strategies. "
                    " - 'rectified': Select every non-zero-selection-"
                    "probability strategy available to each player.")

# General (RL) agent parameters
flags.DEFINE_string("oracle_type", "BR", "Choices are DQN, PG (Policy "
                    "Gradient) or BR (exact Best Response)")
flags.DEFINE_integer("number_training_episodes", int(1e4), "Number training "
                     "episodes per RL policy. Used for PG and DQN")
flags.DEFINE_float("self_play_proportion", 0.0, "Self play proportion")
flags.DEFINE_integer("hidden_layer_size", 256, "Hidden layer size")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_float("sigma", 0.0, "Policy copy noise (Gaussian Dropout term).")
flags.DEFINE_string("optimizer_str", "adam", "'adam' or 'sgd'")

# Policy Gradient Oracle related
flags.DEFINE_string("loss_str", "qpg", "Name of loss used for BR training.")
flags.DEFINE_integer("num_q_before_pi", 8, "# critic updates before Pi update")
flags.DEFINE_integer("n_hidden_layers", 4, "# of hidden layers")
flags.DEFINE_float("entropy_cost", 0.001, "Self play proportion")
flags.DEFINE_float("critic_learning_rate", 1e-2, "Critic learning rate")
flags.DEFINE_float("pi_learning_rate", 1e-3, "Policy learning rate.")

# DQN
flags.DEFINE_float("dqn_learning_rate", 1e-2, "DQN learning rate.")
flags.DEFINE_integer("update_target_network_every", 1000, "Update target "
                     "network every [X] steps")
flags.DEFINE_integer("learn_every", 10, "Learn every [X] steps.")

# General
flags.DEFINE_integer("seed", 1, "Seed.")
flags.DEFINE_bool("local_launch", False, "Launch locally or not.")
flags.DEFINE_bool("verbose", True, "Enables verbose printing and profiling.")
flags.DEFINE_bool("training", False, "Whether Training or Testing")

#NeuPL
flags.DEFINE_integer("N", 4, "policy population size")

def init_pg_responder(sess, env):
  """Initializes the Policy Gradient-based responder and agents."""
  info_state_size = env.observation_spec()["info_state"][0] #for one player 
  num_actions = env.action_spec()["num_actions"]            #how many possible actions

  #takes PG *agent* from algorithms.policy_gradient (where code to update networks lives!!!), 
  #generates *agent class* for policy gradient agent 
  agent_class = rl_policy.PGPolicy   #https://github.com/lucyhalperin/open_spiel_acl/blob/master/open_spiel/python/algorithms/psro_v2/rl_policy.py

  agent_kwargs = {
      "session": sess,
      "info_state_size": info_state_size,
      "num_actions": num_actions,
      "N": FLAGS.N,
      "loss_str": FLAGS.loss_str,
      "loss_class": False,
      "hidden_layers_sizes": [FLAGS.hidden_layer_size] * FLAGS.n_hidden_layers,
      "batch_size": FLAGS.batch_size,
      "entropy_cost": FLAGS.entropy_cost,
      "critic_learning_rate": FLAGS.critic_learning_rate,
      "pi_learning_rate": FLAGS.pi_learning_rate,
      "num_critic_before_pi": FLAGS.num_q_before_pi,
      "optimizer_str": FLAGS.optimizer_str
  }
  oracle = rl_oracle.RLOracle(  #oracle computes best response!
      env,
      agent_class,
      agent_kwargs,
      number_training_episodes=FLAGS.number_training_episodes,
      self_play_proportion=FLAGS.self_play_proportion,
      sigma=FLAGS.sigma)

  agents = [                   #rl_policy.PGPolicy agents with  assigned IDs (2 agents for 2 player game)
      agent_class(  # pylint: disable=g-complex-comprehension
          env,
          player_id,
          **agent_kwargs)
      for player_id in range(FLAGS.n_players)
  ]
  for agent in agents: #preventing any training to take place through calls to the step function
    agent.freeze()
  return oracle, agents

def init_br_responder(env):
  """Initializes the tabular best-response based responder and agents."""
  random_policy = policy.TabularPolicy(env.game) #i.e., not network policy (table of state action conbos)
  oracle = best_response_oracle.BestResponseOracle( #exact best response
      game=env.game, policy=random_policy)
  agents = [random_policy.__copy__() for _ in range(FLAGS.n_players)]
  return oracle, agents


def init_dqn_responder(sess, env):
  """Initializes the Policy Gradient-based responder and agents."""
  state_representation_size = env.observation_spec()["info_state"][0] 
  num_actions = env.action_spec()["num_actions"]

  agent_class = rl_policy.DQNPolicy  #DQN agent class from python.algorithms.dqn
  agent_kwargs = {
      "session": sess,
      "state_representation_size": state_representation_size,
      "num_actions": num_actions,
      "N": FLAGS.N,
      "hidden_layers_sizes": [FLAGS.hidden_layer_size] * FLAGS.n_hidden_layers,
      "batch_size": FLAGS.batch_size,
      "learning_rate": FLAGS.dqn_learning_rate,
      "update_target_network_every": FLAGS.update_target_network_every,
      "learn_every": FLAGS.learn_every,
      "optimizer_str": FLAGS.optimizer_str
  }
  oracle = rl_oracle.RLOracle(
      env,
      agent_class,
      agent_kwargs,
      number_training_episodes=FLAGS.number_training_episodes,
      self_play_proportion=FLAGS.self_play_proportion,
      sigma=FLAGS.sigma)

  agents = [
      agent_class(  # pylint: disable=g-complex-comprehension
          env,
          player_id,
          **agent_kwargs)
      for player_id in range(FLAGS.n_players)
  ]
  for agent in agents:
    agent.freeze()
  return oracle, agents 


def print_policy_analysis(policies, game, verbose=False):
  """Function printing policy diversity within game's known policies.

  Warning : only works with deterministic policies.
  Args:
    policies: List of list of policies (One list per game player)
    game: OpenSpiel game object.
    verbose: Whether to print policy diversity information. (True : print)

  Returns:
    List of list of unique policies (One list per player)
  """
  states_dict = get_all_states.get_all_states(game, np.infty, False, False)
  unique_policies = []
  for player in range(len(policies)):
    cur_policies = policies[player]
    cur_set = set()
    for pol in cur_policies:
      cur_str = ""
      for state_str in states_dict:
        if states_dict[state_str].current_player() == player:
          pol_action_dict = pol(states_dict[state_str])
          max_prob = max(list(pol_action_dict.values()))
          max_prob_actions = [
              a for a in pol_action_dict if pol_action_dict[a] == max_prob
          ]
          cur_str += "__" + state_str
          for a in max_prob_actions:
            cur_str += "-" + str(a)
      cur_set.add(cur_str)
    unique_policies.append(cur_set)
  if verbose:
    print("\n=====================================\nPolicy Diversity :")
    for player, cur_set in enumerate(unique_policies):
      print("Player {} : {} unique policies.".format(player, len(cur_set)))
  print("")
  return unique_policies

def gpsro_looper(env, oracle, agents):
  """Initializes and executes the GPSRO training loop."""
  sample_from_marginals = True  # TODO(somidshafiei) set False for alpharank #???
  training_strategy_selector = FLAGS.training_strategy_selector or strategy_selectors.probabilistic #???

  #initialize psro solver
  g_psro_solver = psro_v2.PSROSolver(  #main PSRO algo!  
      env.game,  
      oracle, # defined via init_XX_responder
      N=FLAGS.N,
      initial_policies=agents,  #list of initial policies for each player
      training_strategy_selector=training_strategy_selector,#probablistic is typical psro
      rectifier=FLAGS.rectifier, #"None": Train against potentially all strategies; "rectified": Train only against strategies beaten by current strat
      sims_per_entry=FLAGS.sims_per_entry, #Number of simulations to run to estimate each element of the game outcome matrix
      number_policies_selected=FLAGS.number_policies_selected, #Number of new strategies trained at each PSRO iteration
      meta_strategy_method=FLAGS.meta_strategy_method, #options: ['alpharank', 'uniform', 'nash', 'prd']
      prd_iterations=50000,
      prd_gamma=1e-10,
      sample_from_marginals=sample_from_marginals,
      symmetric_game=FLAGS.symmetric_game)

  start_time = time.time()
  for gpsro_iteration in range(FLAGS.gpsro_iterations): #Number of training steps for GPSRO
    if FLAGS.verbose:
      print("Iteration : {}".format(gpsro_iteration))
      print("Time so far: {}".format(time.time() - start_time))

    g_psro_solver.iteration() #in AbstractMetaTrainer class (Generate new BR agents via oracle, update gamestate matrix, compute meta strategy)
    meta_game = g_psro_solver.get_meta_game()
    meta_probabilities = g_psro_solver.get_meta_strategies()
    policies = g_psro_solver.get_policies()
    
    if FLAGS.verbose:
      print("Meta game : {}".format(meta_game))
      print("Probabilities : {}".format(meta_probabilities))

    # The following lines only work for sequential games for the moment.
    if env.game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL: #if sequential game
      aggregator = policy_aggregator.PolicyAggregator(env.game) #create aggregator object
      aggr_policies = aggregator.aggregate( 
          range(FLAGS.n_players), [[policies[0]],[policies[1]]], meta_probabilities)      
      
      exploitabilities, expl_per_player = exploitability.nash_conv(  #calculate exploitability
          env.game, aggr_policies, return_only_nash_conv=False)
    
      if FLAGS.verbose:
        print("Exploitabilities : {}".format(exploitabilities))
        print("Exploitabilities per player : {}".format(expl_per_player))
        
        ##PLOTTING FOR DEBUGGING
        if gpsro_iteration%10 == 0:
          ax = plt.gca()
          img = ax.imshow(meta_probabilities, extent=[0,4,4,0], cmap = "YlGnBu",aspect="auto")
          labels = ["1","2","3","4"]
          plt.xticks([0.5,1.5,2.5,3.5])
          plt.yticks([0.5,1.5,2.5,3.5])
          ax.set_xticklabels(labels)
          ax.set_yticklabels(labels)
          plt.colorbar(img)
          plt.savefig("./Results/MetaProb_iteration_" + str(gpsro_iteration) + ".eps", dpi=1200)
          plt.clf()

def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  
  #seeding 
  np.random.seed(FLAGS.seed) 
  tf.set_random_seed(FLAGS.seed)

  game = pyspiel.load_game_as_turn_based(FLAGS.game_name) #game specific info, "game object contains the high level description for a game:                          
  env = rl_environment.Environment(game) #This module wraps Open Spiel Python interface providing an RL-friendly API
  
  #seeding 
  env.seed(FLAGS.seed)
  random.seed(FLAGS.seed)

  # Initialize oracle and agents (DQN, PG, or BR)
  with tf.Session() as sess:
    if FLAGS.training == True:
      if FLAGS.oracle_type == "DQN":
        oracle, agents = init_dqn_responder(sess, env)
      elif FLAGS.oracle_type == "PG":
        oracle, agents = init_pg_responder(sess, env)
      elif FLAGS.oracle_type == "BR":
        oracle, agents = init_br_responder(env)
      sess.run(tf.global_variables_initializer()) 
      gpsro_looper(env, oracle, agents)

    else:
      pass #not testing on this branch yet 

if __name__ == "__main__":
  app.run(main)
