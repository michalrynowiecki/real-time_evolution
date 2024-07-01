
def calculate_avg_lifetime(env,obs, player_N):
  sum = 0
  for i in range(player_N):
    if i+1 in env.realm.players.entities and i+1 in obs:
      sum += env.realm.players.entities[i+1].__dict__['time_alive'].val
  sum = sum/player_N
  return sum

