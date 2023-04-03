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

"""Play bots against each other."""

import pyspiel
from open_spiel.python.bots.policy import PolicyBot
from open_spiel.python.algorithms.psro_v2.utils import load_object
import random 
import numpy as np

def evaluate_bots(state, bots, rng,NS=False):
  """Plays bots against each other, returns terminal utility for each bot."""
  
  #assign basis_bot policy
  for bot in bots:
    bot.restart_at(state)
  counter = 0 
  while not state.is_terminal():

    #if non-stationary, change basis policy after two steps 
    if NS and counter %2 == 0:
      basis = '9' #42, 18
      bots[1]._policy._policy.restore('./test/kuhn_poker/agent1_' + str(basis) + '/')
    elif NS and counter %2 != 0:
      basis = '18'
      bots[1]._policy._policy.restore('./test/kuhn_poker/agent1_' + str(basis) + '/')

    if state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
      action = rng.choice(outcomes, p=probs)
      for bot in bots:
        bot.inform_action(state, pyspiel.PlayerId.CHANCE, action)
      state.apply_action(action)
    elif state.is_simultaneous_node():

      joint_actions = [
          bot.step(state)
          if state.legal_actions(player_id) else pyspiel.INVALID_ACTION
          for player_id, bot in enumerate(bots)
      ]
      state.apply_actions(joint_actions)
    else:
      current_player = state.current_player()
      action = bots[current_player].step(state)
      for i, bot in enumerate(bots):
        if i != current_player:
          bot.inform_action(state, current_player, action)
      state.apply_action(action)
    counter += 1 
  return state.returns()

