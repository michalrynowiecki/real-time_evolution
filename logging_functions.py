
def calculate_avg_lifetime(env,obs, player_N):
  sum = 0
  for i in range(player_N):
    if i+1 in env.realm.players.entities and i+1 in obs:
      sum += env.realm.players.entities[i+1].__dict__['time_alive'].val
  sum = sum/player_N
  return sum

# Returns the number of agents whose lifetime is > N, where N is the amount of steps an agent can survive without consuming water or food
def above_reg(lifetimes: dict[int, int], N: int) -> int:
  return sum(1 for value in lifetimes.values() if value > N)

# Returns the average exp of agents whose lifetime is above N
def avg_expN(exp_dict: dict, lifetimes: dict[int, int], N: int) -> int:
    total_sum = 0
    for elem in exp_dict:
        if lifetimes.get(elem, 0) > N:
            total_sum += exp_dict[elem]
    return total_sum

# After reset
travelled_dist = {for each agent: val = 0, last_pos = cur_pos}

def migration():
   
if (step == 0):
   