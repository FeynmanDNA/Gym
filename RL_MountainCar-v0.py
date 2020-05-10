import gym
import time
import random
import numpy as np


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

# initial q_table
# with zeros
q_table = np.zeros((num_buckets, num_buckets, env.action_space.n))
# or with random values
#  q_table = np.random.uniform(low=-2, high=0, size=(num_buckets, num_buckets, env.action_space.n))

print("initial q_table: ", q_table, q_table.shape)

print("initial discrete observation state in q_table: ", q_table[discrete_observation_state])

print("initial action based on the q_table: ", np.argmax(q_table[discrete_observation_state]))

#********************* main program **********************

# @hyperparameters
num_episodes = 10000
max_steps_per_episode = 200

learning_rate = 0.1
discount_rate = 0.99

# Exploration parameters
exploration_rate = 1
max_exploration_rate = exploration_rate
min_exploration_rate = 0.01
exploration_decay_rate = 0.001
# decay the exploration_rate after which episode
start_decay = 1500

show_every = 500
# metric for evaluation
rewards_all_episodes = []

# Q-learning algorithm
for episode in range(num_episodes):
    # only render the game every once a while
    if episode % show_every == 0:
        render = False
        np.save(f"{episode}-qtable.npy", q_table)
    else:
        render = False

    # initialize new episode params
    state = env.reset()
    # reset the discrete_observation_state as well!
    discrete_observation_state = get_discrete_observation_state(state)
    done = False

    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        #  print("q_table: ")
        #  print(np.around(q_table, decimals=3))

        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        ## If this number > greater than epsilon --> exploitation 
        #(taking the biggest Q value for this state)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[discrete_observation_state])
        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)
        # print(f"new_state: {new_state}, reward: {reward}, done: {done}, info: {info}")

        # new discrete observed state for calculating q value
        new_discrete_observation_state = get_discrete_observation_state(new_state)

        # Update Q-table for Q(s,a)
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        max_future_q = np.max(q_table[new_discrete_observation_state])
        current_q = q_table[discrete_observation_state+(action,)]
        new_q = current_q * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * max_future_q)
        # update the q_table with new_q
        q_table[discrete_observation_state+(action,)] = new_q
        # NOTE: we are updating the q_table for the action
        # AFTER we have taken that action and received the reward and new state

        # Set new state
        discrete_observation_state = new_discrete_observation_state

        # Add new reward
        rewards_current_episode += reward

        if render:
            # env.render() to display the graphics
            env.render()
            #  print()
            # pause to see the updates
            #  time.sleep(0.01)
        #  clear_output(wait=True)
        # check if the last action ended the episode
        if done == True and new_state[0] >= env.goal_position:
            # reward is only -1
            # -1 for each time step, until the goal position of 0.5 is reached
            #  q_table[discrete_observation_state+(action,)] = 0
            break

    # Exploration rate decay
    # Reduce epsilon (because we need less and less exploration)
    if episode >= start_decay:
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*(episode-start_decay))

    # Add current episode reward to total rewards list
    rewards_all_episodes.append(rewards_current_episode)

    print("Episode {}, episode reward {}, epsilon: {}".format(episode, rewards_current_episode, exploration_rate))
    # move on to the next episode

# Calculate and print the average reward per show_every episodes
rewards_per_N_episodes = np.split(np.array(rewards_all_episodes),num_episodes/show_every)
count = show_every

# for plotting
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

print("\n********Stats per {} episodes********\n".format(show_every))
for r in rewards_per_N_episodes:
    print(count, "avg: ", str(sum(r/show_every)))
    print(count, "min: ", str(min(r)))
    print(count, "max: ", str(max(r)))

    aggr_ep_rewards['ep'].append(count)
    aggr_ep_rewards['avg'].append(sum(r/show_every))
    aggr_ep_rewards['min'].append(min(r))
    aggr_ep_rewards['max'].append(max(r))

    count += show_every

import matplotlib.pyplot as plt

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=2)
title = "{} Episodes, {:.2f} LR, {:.4f} explore rate, {:.4f} Decay rate\n {} start decay, {} q_table shape, init Zeros".format(num_episodes, learning_rate, max_exploration_rate, exploration_decay_rate, start_decay, q_table.shape)
print("Title: ", title)
plt.title(title)
plt.grid(True)
plt.show()

env.close()
