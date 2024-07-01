import torch
import numpy as np
import random
from math import ceil, sqrt

#Checking for how long each of the agents has travelled during the course of its life
def get_distance_travelled(entities, spawn_positions, agent_number):
  x_2, y_2 = entities[agent_number].pos #current position coordinates
  x_1, y_1 = spawn_positions[agent_number] #spawn position coordinates

  dist_squared = (x_2 - x_1)**2 + (y_2 - y_1)**2
  if (dist_squared) != 0:
    return sqrt(dist_squared)
  else:
    return 0



def fitness(env, player_N):
  fitness_dict = {}

  # Calculate distance travelled for each agent
  for i in range(player_N):
    if i+1 in env.realm.players.entities:
      fitness_dict[i+1] = get_distance_travelled(env.realm.players.entities, env.game_state.spawn_pos, i+1)

  # Get index of max value (the best agent number)
  best_agent = max(fitness_dict, key=fitness_dict.get)

  return fitness_dict

#Select top beta % of agents that travelled the longest distances/lived the longest/both and then from those randomly pick the parent
def pick_best(env, player_N, life_durations, top=0.2):
  beta = ceil(top * env.config.PLAYER_N)

  # Get beta top percent of both
  my_keys_long_runners = sorted(fitness(env, player_N), key=fitness(env, player_N).get, reverse=True)[:beta]
  my_keys_long_livers = sorted(life_durations, key=life_durations.get, reverse=True)[:beta]
  bestest = list(set(my_keys_long_runners).intersection(my_keys_long_livers))

  # Pick the best from the intersetction of the longest living agents and furthest walking agents. If that is empty then pick one from the longest living agents.
  if bestest:
    parent = random.choice(bestest)
  else:
    #parent = random.choice(my_keys_long_livers)
    parent = random.choice(my_keys_long_runners)
  return parent


# TODO: Make this more efficient - spatial locality
def mutate(player_num, parent, model_dict, life_durations, alpha=0.01, dynamic_alpha=True):

  if dynamic_alpha:
    ##dynamic learning rate based on parent lifetime
    ##TODO try continuous learning rate
    if life_durations[parent] > 60:
      alpha = 0.01
    elif life_durations[parent] > 30:
      alpha = 0.025
    else:
      alpha = 0.075

  # mutate movement network
  for param in model_dict[player_num].parameters():
    with torch.no_grad():
      param.add_(torch.randn(param.size()) * alpha)


