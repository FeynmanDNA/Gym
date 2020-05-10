import gym
import numpy as np
import random
import time


# discrete action space example with MountainCar-v0
env_name = "MountainCar-v0"
# continuous action space with MountainCarContinuous-v0
# env_name = "MountainCarContinuous-v0"
env = gym.make(env_name)

# observation space
print("Observation space: ", env.observation_space)
print("observation space high, low: ", env.observation_space.high, env.observation_space.low)
# MontainCar observation is [position, velocity]
# -1.2<position <0.6
# -0.07<position <0.07
# high for possible observation is [0.6 0.07]
# low for possible observation is [-1.2 -0.07]

# action space
print("Action space: ", env.action_space)

# initial state
state = env.reset()

# bucket for (position, velocity) observation space
num_buckets = 20
observation_bucket_size = (env.observation_space.high - env.observation_space.low) / num_buckets

print("bucket size: ", observation_bucket_size)

# discrete observation_space
def get_discrete_observation_state(state):
    """state is a tuple of (pos, vel)"""
    discrete_observation = (state - env.observation_space.low) / observation_bucket_size
    discrete_observation_int = tuple(discrete_observation.astype(np.int))
    return discrete_observation_int

discrete_observation_state = get_discrete_observation_state(state)
print("initial state: ", state)
print("initial discrete observation state:", discrete_observation_state)

# load the q_table with episode index
i = 8500
q_table = np.load(f"MountainCarv0-{i}-qtable.npy")

# Print updated Q-table
print("\n\n********Q-table********\n")
#  print(q_table)

print("initial discrete observation state in q_table: ", q_table[discrete_observation_state])

print("initial action based on the q_table: ", np.argmax(q_table[discrete_observation_state]))

max_steps_per_episode = 200

for episode in range(3):
    # initialize new episode params
    state = env.reset()
    # reset the discrete_observation_state as well!
    discrete_observation_state = get_discrete_observation_state(state)
    done = False
    rewards_current_episode = 0
    print("\n*****EPISODE ", episode+1, "*****\n")
    # time.sleep(1)

    for step in range(max_steps_per_episode):
        # Show current state of environment on screen
        env.render()

        # Choose action with highest Q-value for current state
        action = np.argmax(q_table[discrete_observation_state])
        # Take new action
        new_state, reward, done, info = env.step(action)

        # Add new reward
        rewards_current_episode += reward

        if done:
            if reward == 0:
                # Agent reached the goal and won episode
                print("****Step: {}, Reward: {}****".format(step, rewards_current_episode))
            else:
                print("****Step: {}, Reward: {}****".format(step, rewards_current_episode))
            break

        # if not done, set new state
        # new discrete observed state
        new_discrete_observation_state = get_discrete_observation_state(new_state)
        discrete_observation_state = new_discrete_observation_state

env.close()
