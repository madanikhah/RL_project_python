import gym
env = gym.make("FrozenLake-v1", render_mode="ansi")

# print(env.observation_space)
# print(env.action_space)
# print(env.P)

env.reset()
print(env.render())

# env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False )
# env.reset()
# env.render()

# 0: left
# 1: down
# 2: right
# 3: up

# action = 2
# env.step(action)
# print(env.render())

# for i in range(10):
#     print(env.render())
#     action = 2
#     env.step(action)

# for i in range(10):
#     print(env.render())
#     action = env.action_space.sample()
#     env.step(action)
    
    
for i in range(10):
    print(env.render())
    action = env.action_space.sample()
    next_state, reward, done, trunc, info = env.step(action)
    print(next_state, reward, done, trunc, info)
    
    if done == True:
        env.reset()