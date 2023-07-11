# import gym

# from gym import Env
import gym
from RL_project import every_visit_mc_prediction, first_visit_mc_prediction, policy_iteration, custom_map_3, custom_map_4,\
    custom_map_5, custom_map_6, custom_map_7, custom_map_1, custom_map_2


# سوال اول

# env = gym.make("FrozenLake-v1", desc=custom_map_1, is_slippery=False)

# hole_reward = -0.1
# goal_reward = 1
# move_reward = -0.1
# theta = 0.0001

# for gamma in [1, 0.9, 0.5, 0.1]:
#     print(f"Gamma: {gamma}")
#     policy, V = policy_iteration(env, custom_map_1, 30, 0.0001, gamma)
#     print("Optimal policy:")
#     print(policy)
#     print("State values:")
#     print(V)



# سوال دوم

# env = gym.make("FrozenLake-v1", desc=custom_map_2, is_slippery=False)

# hole_reward=-4
# goal_reward = 10
# move_reward = -0.9
# theta = 0.0001

# for gamma in [1, 0.9, 0.5, 0.1]:
#     print(f"Gamma: {gamma}")
#     policy, V = policy_iteration(env, custom_map_2, 30, 0.0001, gamma)
#     print("Optimal policy:")
#     print(policy)
#     print("State values:")
#     print(V)



# سوال سوم


# env = gym.make("FrozenLake-v1", desc=custom_map_3, is_slippery=True)


# hole_reward=-5
# goal_reward = 5
# move_reward = -0.5
# theta = 0.0001



# for gamma in [0.9]:
#     print(f"Gamma: {gamma}")
#     policy, V = policy_iteration(env, custom_map_3, 30, 0.0001, gamma)
#     print("Optimal policy:")
#     print(policy)
#     print("State values:")
#     print(V)



# سوال چهارم

# env = gym.make("FrozenLake-v1", desc=custom_map_4, is_slippery=True)

# hole_reward=-5
# goal_reward = 5
# move_reward = -0.5
# theta = 0.0001

# for gamma in [0.9]:
#     print(f"Gamma: {gamma}")
#     policy, V = policy_iteration(env, custom_map_4, 30, 0.0001, gamma)
#     print("Optimal policy:")
#     print(policy)
#     print("State values:")
#     print(V)



# # سوال پنجم

# env = gym.make("FrozenLake-v1", desc=custom_map_5, is_slippery=False)


# hole_reward=-3
# goal_reward = 7
# # move_reward = -0.5
# theta = 0.0001
# gamma=0.9



# for move_reward in [7]:
#     print(f"move_reward: {move_reward}")
#     policy, V = policy_iteration(env, custom_map_5, 30, theta, gamma)
#     print("Optimal policy:")
#     print(policy)
#     print("State values:")
#     print(V)



# سوال ششم

# env = gym.make("FrozenLake-v1", desc=custom_map_6, is_slippery=True)


# hole_reward=-3
# goal_reward = 7
# # move_reward = -0.5
# theta = 0.0001
# gamma=0.9



# for move_reward in [-4,-2,0,2]:
#     print(f"move_reward: {move_reward}")
#     policy, V = policy_iteration(env, custom_map_6, 30, theta, gamma)
#     print("Optimal policy:")
#     print(policy)
#     print("State values:")
#     print(V)



# سوال هفتم

env = gym.make("FrozenLake-v1", desc=custom_map_7, is_slippery=True)


hole_reward=-2
goal_reward = 50
# move_reward = -1
theta = 0.0001
gamma=0.9



for move_reward in [-1]:
    print(f"move_reward: {move_reward}")
    policy, V = policy_iteration(env, custom_map_7, 30, theta, gamma)
    print("Optimal policy:")
    print(policy)
    print("State values:")
    print(V)

for num_episodes in [500, 5000]:
    V_first_visit = first_visit_mc_prediction(env, policy, num_episodes, gamma)
    print("State values first visit:")
    print(V_first_visit)


for num_episodes in [500, 5000]:
    V_every_visit = every_visit_mc_prediction(env, policy, num_episodes, gamma)
    print("State values every visit:")
    print(V_every_visit)

