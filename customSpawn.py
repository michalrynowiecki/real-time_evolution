# Keep an array where at each index we track how long each agent has been dead for.
# If this value exceeds N, spawn the agent

def delayedSpawn(deadArray, playerN, threshold):
    if deadArray[playerN] > threshold:
        
        parent = pick_best(env, player_N, life_durations, spawn_positions,)
        try:
          x, y = env.realm.players.entities[parent].pos
        except:
          x, y = random.choice(spawn_positions)
        
        spawn_positions[i+1] = (x,y)

        env.realm.players.spawn_individual(x, y, i+1)

        life_durations[i+1] = 0
        model_dict[i+1] = copy.deepcopy(model_dict[parent])
        model_dict[i+1].hidden = (torch.zeros(model_dict[i+1].hidden[0].shape), torch.zeros(model_dict[i+1].hidden[1].shape))
        mutate(i+1, parent, model_dict, life_durations, alpha=0.02, dynamic_alpha=True)

    else:
       deadArray[playerN] += 1