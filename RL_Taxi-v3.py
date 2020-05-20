import numpy as np
import gym
import time
import random


env = gym.make("Taxi-v3")

env.render()

is_discrete = (type(env.action_space) == gym.spaces.discrete.Discrete)

if is_discrete:
    # action_space.n is for discrete action space
    action_size = env.action_space.n
    print("Action size:", action_size)
else:
    # action_space.low and action_space.high
    action_low = env.action_space.low
    action_high = env.action_space.high
    action_shape = env.action_space.shape
    print("Action range:", action_low, action_high)

state_size = env.observation_space.n
print("State size: ", state_size)

# q_table rows = state, columns = actions
q_table = np.zeros((state_size, action_size))
print("init q_table: ", q_table)

# hyperparameters
total_episodes = 2000
total_test_episodes = 100
max_steps = 99  # max steps per episode

learning_rate = 0.7
gamma = 0.618  # discounting rate

# exploration paramters
epsilon = 1.0  # exploration rate
max_epsilon = 1.0  # starting exploration probability
min_epsilon = 0.01 # minimum exploration probability
decay_rate = 0.01  # exponential decay rate for exploration prob


# **************** q learning *****************

for episode in range(total_episodes):
    print("\nNew Episode ", episode)
    # reset the environment
    state = env.reset()
    done = False
    rewards_current_episode = 0

    for step in range(max_steps):
        # choose an action in the current state
        if random.uniform(0,1) > epsilon:
            # eploitation
            action = np.argmax(q_table[state, :])
        else:
            # exploration
            action = env.action_space.sample()

        # take the action and observe the outcome
        new_state, reward, done, info = env.step(action)

        # update Q table of the last state
        q_table[state, action] = q_table[state, action] +\
            learning_rate * (reward + \
                gamma * np.max(q_table[new_state, :]) -\
                q_table[state, action])

        # Add new reward
        rewards_current_episode += reward
        # reset state to the observed state
        state = new_state

        print("Episode {}, episode reward {}, epsilon: {}".format(episode, rewards_current_episode, epsilon))
        #  env.render()
        #  print()
        # pause to see the updates
        #  time.sleep(1)

        # if done, finish episode
        if done is True:
            break

    # reduce epsilon
    epsilon = min_epsilon + (max_epsilon-min_epsilon)*\
            np.exp(-decay_rate*episode)


# save the last q_table
print("last q_table: ")
print(q_table)
np.save("RL_Taxi-v3_Qtable.npy", q_table)

env.close()

