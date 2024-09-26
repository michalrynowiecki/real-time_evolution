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



def fitness(env, player_N, spawn_positions):
  fitness_dict = {}

  # Calculate distance travelled for each agent
  for i in range(player_N):
    if i+1 in env.realm.players.entities:
      fitness_dict[i+1] = get_distance_travelled(env.realm.players.entities, spawn_positions, i+1)

  # Get index of max value (the best agent number)
  best_agent = max(fitness_dict, key=fitness_dict.get)

  return fitness_dict

#Select top beta % of agents that travelled the longest distances/lived the longest/both and then from those randomly pick the parent
def pick_best(env, player_N, life_durations, spawn_positions, top=0.1):
  beta = ceil(top * env.config.PLAYER_N)

  #'''
  melee_dict = {}
  range_dict = {}
  mage_dict = {}
  fish_dict = {}
  herb_dict = {}
  prosp_dict = {}
  carv_dict = {}
  alch_dict = {}

  melee_sum = 0
  range_sum = 0
  mage_sum = 0
  fish_sum = 0
  herb_sum = 0
  prosp_sum = 0
  carv_sum = 0
  alch_sum = 0


  #for i in range(player_N):
  for player in env.realm.players.entities:
    #i = env.realm.players.entities[i+1].id.val - 1
    i = env.realm.players.entities[player].id.val

    melee_dict[i] = env.realm.players.entities[i].melee_exp.val
    melee_sum += env.realm.players.entities[i].melee_exp.val

    range_dict[i] = env.realm.players.entities[i].range_exp.val
    range_sum += env.realm.players.entities[i].range_exp.val

    mage_dict[i] = env.realm.players.entities[i].mage_exp.val
    mage_sum += env.realm.players.entities[i].mage_exp.val

    fish_dict[i] = env.realm.players.entities[i].fishing_exp.val
    fish_sum += env.realm.players.entities[i].fishing_exp.val

    herb_dict[i] = env.realm.players.entities[i].herbalism_exp.val
    herb_sum += env.realm.players.entities[i].herbalism_exp.val

    prosp_dict[i] = env.realm.players.entities[i].prospecting_exp.val
    prosp_sum += env.realm.players.entities[i].prospecting_exp.val

    carv_dict[i] = env.realm.players.entities[i].carving_exp.val
    carv_sum += env.realm.players.entities[i].carving_exp.val

    alch_dict[i] = env.realm.players.entities[i].alchemy_exp.val
    alch_sum += env.realm.players.entities[i].alchemy_exp.val

  if melee_sum > -1:
    top_melee = sorted(melee_dict, key=melee_dict.get, reverse=True)[:beta]
  else:
    top_melee = []

  if range_sum > -1:
    top_range = sorted(range_dict, key=range_dict.get, reverse=True)[:beta]
  else:
    top_range = []

  if mage_sum > -1:
    top_mage = sorted(mage_dict, key=mage_dict.get, reverse=True)[:beta]
  else:
    top_mage = []

  if fish_sum > -1:
    top_fish = sorted(fish_dict, key=fish_dict.get, reverse=True)[:beta]
  else:
    top_fish = []

  if herb_sum > -1:
    top_herb = sorted(herb_dict, key=herb_dict.get, reverse=True)[:beta]
  else:
    top_herb = []

  if prosp_sum > -1:
    top_prosp = sorted(prosp_dict, key=prosp_dict.get, reverse=True)[:beta]
  else:
    top_prosp = []

  if carv_sum > -1:
    top_carv = sorted(carv_dict, key=carv_dict.get, reverse=True)[:beta]
  else:
    top_carv = []

  if alch_sum > -1:
    top_alch = sorted(alch_dict, key=alch_dict.get, reverse=True)[:beta]
  else:
    top_alch = []
  #'''
  # Get beta top percent of both
  top_runners = sorted(fitness(env, player_N, spawn_positions), key=fitness(env, player_N, spawn_positions).get, reverse=True)[:beta]
  top_livers = sorted(life_durations, key=life_durations.get, reverse=True)[:beta]


  #pool =  top_livers + top_runners + top_melee + top_range + top_mage + top_fish + top_herb + top_prosp + top_carv + top_alch
  pool =  top_fish + top_herb + top_prosp + top_carv + top_alch

  parent = random.choice(pool)

  return parent



#Select top beta % of agents that travelled the longest distances/lived the longest/both and then from those randomly pick the parent
def pick_best_old(env, player_N, life_durations, spawn_positions, top=0.2):
  beta = ceil(top * env.config.PLAYER_N)

  # Get beta top percent of both
  my_keys_long_runners = sorted(fitness(env, player_N, spawn_positions), key=fitness(env, player_N, spawn_positions).get, reverse=True)[:beta]
  my_keys_long_livers = sorted(life_durations, key=life_durations.get, reverse=True)[:beta]

  bestest = my_keys_long_runners + my_keys_long_livers  #list(set(my_keys_long_runners).intersection(my_keys_long_livers))

  # Pick the best from the intersetction of the longest living agents and furthest walking agents. If that is empty then pick one from the longest living agents.
  if bestest:
    parent = random.choice(bestest)
  else:
    parent = random.choice(my_keys_long_livers)
    #parent = random.choice(my_keys_long_runners)
  return parent


# TODO: Make this more efficient - spatial locality
def mutate(player_num, parent, model_dict, life_durations, alpha=0.01, dynamic_alpha=True):

  if dynamic_alpha:
    ##dynamic learning rate based on parent lifetime
    ##TODO try continuous learning rate
    if life_durations[parent] > 60:
      alpha = 0.001
    elif life_durations[parent] > 30:
      alpha = 0.0025
    else:
      alpha = 0.0075

  # mutate movement network
  for param in model_dict[player_num].parameters():
    with torch.no_grad():
      param.add_(torch.randn(param.size()) * alpha)

import copy
def delayedSpawn(env, life_durations, model_dict, spawn_positions, deadDict, player_N, threshold):
    if deadDict[player_N] > threshold:
        parent = pick_best(env, player_N, life_durations, spawn_positions,)
        try:
          x, y = env.realm.players.entities[parent].pos
        except:
          x, y = random.choice(spawn_positions)
        
        spawn_positions[player_N+1] = (x,y)

        env.realm.players.spawn_individual(x, y, player_N+1)

        life_durations[player_N+1] = 0
        model_dict[player_N+1] = copy.deepcopy(model_dict[parent])
        model_dict[player_N+1].hidden = (torch.zeros(model_dict[player_N+1].hidden[0].shape), torch.zeros(model_dict[player_N+1].hidden[1].shape))
        mutate(player_N+1, parent, model_dict, life_durations, alpha=0.02, dynamic_alpha=True)

    else:
       deadDict[player_N] += 1


def fitness(env, player_N, spawn_positions):
  fitness_dict = {}

  # Calculate distance travelled for each agent
  for i in range(player_N):
    if i+1 in env.realm.players.entities:
      fitness_dict[i+1] = get_distance_travelled(env.realm.players.entities, spawn_positions, i+1)

  # Get index of max value (the best agent number)
  best_agent = max(fitness_dict, key=fitness_dict.get)

  return fitness_dict

#Select top beta % of agents that travelled the longest distances/lived the longest/both and then from those randomly pick the parent
def pick_best_logprob(env, player_N, life_durations, spawn_positions, top=0.1):

  # Create a dictionary where each entry is the different exp
  exp_dict = {i: {
    "melee":  2,
    "range":  2,
    "mage":   2,
    "fish":   2,
    "herb":   2,
    "prosp":  2,
    "carv":   2,
    "alch":   2
  } for i in env.realm.players.entities}
  

  melee_sum = 0.01
  range_sum = 0.01
  mage_sum = 0.01
  fish_sum = 0.01
  herb_sum = 0.01
  prosp_sum = 0.01
  carv_sum = 0.01
  alch_sum = 0.01


  # for i in range(player_N):
  for player in env.realm.players.entities:
    i = env.realm.players.entities[player].id.val
    #i = env.realm.players.entities[i+1].id.val - 1

    exp_dict[i]["melee"] = env.realm.players.entities[i].melee_exp.val
    melee_sum += env.realm.players.entities[i].melee_exp.val

    exp_dict[i]["range"] = env.realm.players.entities[i].range_exp.val
    range_sum += env.realm.players.entities[i].range_exp.val

    exp_dict[i]["mage"] = env.realm.players.entities[i].mage_exp.val
    mage_sum += env.realm.players.entities[i].mage_exp.val

    exp_dict[i]["fishing"] = env.realm.players.entities[i].fishing_exp.val
    fish_sum += env.realm.players.entities[i].fishing_exp.val

    exp_dict[i]["herbalism"] = env.realm.players.entities[i].herbalism_exp.val
    herb_sum += env.realm.players.entities[i].herbalism_exp.val

    exp_dict[i]["prosp"] = env.realm.players.entities[i].prospecting_exp.val
    prosp_sum += env.realm.players.entities[i].prospecting_exp.val

    exp_dict[i]["carv"] = env.realm.players.entities[i].carving_exp.val
    carv_sum += env.realm.players.entities[i].carving_exp.val

    exp_dict[i]["alch"] = env.realm.players.entities[i].alchemy_exp.val
    alch_sum += env.realm.players.entities[i].alchemy_exp.val

  # For each of the players, assign the highest exp value
  specialized_exp = {i: max(player_exp.values()) for i, player_exp in exp_dict.items()}
  print(specialized_exp)
  from math import log
  total_exp = melee_sum + range_sum + mage_sum + fish_sum + herb_sum + prosp_sum + carv_sum + alch_sum
  print("total exp: ", total_exp)
  from math import log
  
  total = sum(specialized_exp.values())

  final_probs_dict = {key: log(value, 2) for key, value in specialized_exp.items()}

  normalized_dict = {key: value / total for key, value in final_probs_dict.items()}
  print(normalized_dict)

  
  items = list(normalized_dict.keys())
  probabilities = list(normalized_dict.values())
  parent = random.choices(items, weights=probabilities, k=1)[0]
  
  return parent