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

"""An Oracle for any RL algorithm.

An Oracle for any RL algorithm following the OpenSpiel Policy API.
"""

import numpy as np

from open_spiel.python.algorithms.psro_v2 import optimization_oracle
from open_spiel.python.algorithms.psro_v2 import utils


def update_episodes_per_oracles(episodes_per_oracle, played_policies_indexes):
  """Updates the current episode count per policy.

  Args:
    episodes_per_oracle: List of list of number of episodes played per policy.
      One list per player.
    played_policies_indexes: List with structure (player_index, policy_index) of
      played policies whose count needs updating.

  Returns:
    Updated count.
  """
  for player_index, policy_index in played_policies_indexes:
    episodes_per_oracle[player_index][policy_index] += 1
  return episodes_per_oracle


def freeze_all(policies_per_player):
  """Freezes all policies within policy_per_player.

  Args:
    policies_per_player: List of list of number of policies.
  """
  for policies in policies_per_player:
    policies.freeze()


def random_count_weighted_choice(count_weight):
  """Returns a randomly sampled index i with P ~ 1 / (count_weight[i] + 1).

  Allows random sampling to prioritize indexes that haven't been sampled as many
  times as others.

  Args:
    count_weight: A list of counts to sample an index from.

  Returns:
    Randomly-sampled index.
  """
  indexes = list(range(len(count_weight)))
  p = np.array([1 / (weight + 1) for weight in count_weight])
  p /= np.sum(p)
  chosen_index = np.random.choice(indexes, p=p)
  return chosen_index


class RLOracle(optimization_oracle.AbstractOracle):
  """Oracle handling Approximate Best Responses computation."""

  def __init__(self,
               env,
               best_response_class,
               best_response_kwargs,
               number_training_episodes=1e3,
               self_play_proportion=0.0,
               **kwargs):
    """Init function for the RLOracle.

    Args:
      env: rl_environment instance.
      best_response_class: class of the best response.
      best_response_kwargs: kwargs of the best response.
      number_training_episodes: (Minimal) number of training episodes to run
        each best response through. May be higher for some policies.
      self_play_proportion: Float, between 0 and 1. Defines the probability that
        a non-currently-training player will actually play (one of) its
        currently training strategy (Which will be trained as well).
      **kwargs: kwargs
    """
    self._env = env

    self._best_response_class = best_response_class
    self._best_response_kwargs = best_response_kwargs

    self._self_play_proportion = self_play_proportion
    self._number_training_episodes = number_training_episodes

    super(RLOracle, self).__init__(**kwargs)

  def sample_episode(self, unused_time_step, opponent, is_evaluation=False):
    agents = [ opponent, self.new_policy]

    time_step = self._env.reset()
    counter = 0 
    cumulative_rewards = 0.0
    while not time_step.last():
      counter += 1
      if time_step.is_simultaneous_move():
        action_list = []
        for agent in agents:
          output = agent.step(time_step, is_evaluation=is_evaluation)
          action_list.append(output.action)
        time_step = self._env.step(action_list)

        cumulative_rewards += np.array(time_step.rewards)
      else:
        player_id = time_step.observations["current_player"]

        # is_evaluation is a boolean that, when False, lets policies train. The
        # setting of PSRO requires that all policies be static aside from those
        # being trained by the oracle. is_evaluation could be used to prevent
        # policies from training, yet we have opted for adding frozen attributes
        # that prevents policies from training, for all values of is_evaluation.
        # Since all policies returned by the oracle are frozen before being
        # returned, only currently-trained policies can effectively learn.
        #print(time_step)
        agent_output = agents[player_id].step(
            time_step, is_evaluation=is_evaluation)
        action_list = [agent_output.action]

        time_step = self._env.step(action_list)

        cumulative_rewards += np.array(time_step.rewards)
        #print(time_step)
        #import pdb; pdb.set_trace()

    if not is_evaluation:
      for agent in agents:
        agent.step(time_step)
  
    return cumulative_rewards

  def _has_terminated(self, episodes_per_oracle):
    # The oracle has terminated when all policies have at least trained for
    # self._number_training_episodes. Given the stochastic nature of our
    # training, some policies may have more training episodes than that value.

    return episodes_per_oracle > self._number_training_episodes

  
  def _rollout(self, game, agents, **oracle_specific_execution_kwargs):
    self.sample_episode(None, agents, is_evaluation=False)

  def generate_new_policies(self, current_pol):
    """Generates new policies to be trained into best responses.

    Args:
      training_parameters: list of list of training parameter dictionaries, one
        list per player.

    Returns:
      List of list of the new policies, following the same structure as
      training_parameters.
    """
    
    if isinstance(current_pol, self._best_response_class):
      new_pol = current_pol.copy_with_noise(self._kwargs.get("sigma", 0.0))
    else:
      new_pol = self._best_response_class(self._env, 1,  #player 0?
                                          **self._best_response_kwargs)
      
      new_pol.unfreeze()
    new_pol._policy.player_id = 1
    self.new_policy = new_pol
    return 

  def generate_agents_for_rollout(self, opponent,strategy_sampler, graph,latent):

    #set oracle (self agent) to be have latent = sigma! 
    self.new_policy._policy.set_latent(latent)

    assert not self.new_policy.is_frozen() #make sure oracle is not frozen
    if sum(latent) != 0:
      assert sum(latent) == 1

    #get opponent policy 
    indices = [i for i in range(opponent._policy._N)] 
  
    # Sample interaction graph to set opponent latent variable for the episode
    episode_opp_latent = strategy_sampler([indices],[latent])[0]
    opponent._policy.set_latent(graph[episode_opp_latent])  

    assert opponent.is_frozen()       #make sure opponent is frozen

    return opponent

  def __call__(self,
               graph,
               game,
               pol,
               strategy_sampler=utils.sample_strategy,
               **oracle_specific_execution_kwargs):
    """Call method for oracle, returns best responses against a set of policies.

    Args:
      game: The game on which the optimization process takes place.
      training_parameters: A list of list of dictionaries (One list per player),
        each dictionary containing the following fields :
        - policy : the policy from which to start training.
        - total_policies: A list of all policy.Policy strategies used for
          training, including the one for the current player.int
        - current_player: Integer representing the current player.
        - probabilities_of_playing_policies: A list of arrays representing, per
          player, the probabilities of playing each policy in total_policies for
          the same player.
      strategy_sampler: Callable that samples strategies from total_policies
        using probabilities_of_playing_policies. It only samples one joint
        set of policies for all players. Implemented to be able to take into
        account joint probabilities of action (For Alpharank)
      **oracle_specific_execution_kwargs: Other set of arguments, for
        compatibility purposes. Can for example represent whether to Rectify
        Training or not.

    Returns:
      A list of list, one for each member of training_parameters, of (epsilon)
      best responses.
    """
                           
    episodes_per_oracle = 0
    unique_rows = {}
    self.generate_new_policies(pol) #initalize new policies to LEARN 
    #unique_rows = np.unique(graph, axis=0)
    
    for index in graph: #for each row in interaction graph 
      if tuple(index) not in unique_rows:
        print("training row " + str(index) + " of interaction graph")
        while not self._has_terminated(episodes_per_oracle):    #train new policies
          opponent = self.generate_agents_for_rollout(pol,strategy_sampler, graph, index) #[agent 0 row, agent 1 row]
          self._rollout(game, opponent, **oracle_specific_execution_kwargs)

          episodes_per_oracle += 1

        unique_rows[tuple(index)] = []
        pol = self.new_policy.copy_with_noise(sigma=0.0) #update opponent current policy for training of next index 

      else:
        print("non-unique row")

    # Freeze the new policies to keep their weights static. This allows us to
    # later not have to make the distinction between static and training
    # policies in training iterations.
    second_player = self.new_policy.copy_with_noise(sigma=0.0) #create second policy to use during updating empirical game state 
    second_player._policy.player_id = 0                          
    #second_player._policy._step_counter = self.new_policy._policy._step_counter
    
    new_policies = [second_player,self.new_policy] #TODO: fix this weird syntax fix 
    freeze_all(new_policies)

    
    return new_policies
