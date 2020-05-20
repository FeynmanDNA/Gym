import numpy as np
import gym
import time
import random


env = gym.make("Taxi-v3")

env.render()


# reset the environment
state = env.reset()

q_table = np.load("./RL_Taxi-v3_Qtable.npy")

# Print updated Q-table
print("\n\n********Q-table********\n")
print(q_table)

max_steps = 99  # max steps per episode

for episode in range(3):
    print("\nNew Episode ", episode)
    # reset the environment
    state = env.reset()
    done = False
    rewards_current_episode = 0

    for step in range(max_steps):
        env.render()
        time.sleep(0.5)
        # eploitation
        action = np.argmax(q_table[state, :])
        # take the action and observe the outcome
        new_state, reward, done, info = env.step(action)

        # Add new reward
        rewards_current_episode += reward

        print("****Step: {}, Reward: {}****".format(step, rewards_current_episode))

        # if done, finish episode
        if done is True:
            break

        # if not done, set new state
        state = new_state


env.close()

