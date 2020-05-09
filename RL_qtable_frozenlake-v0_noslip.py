import gym
import numpy as np
import random
import time
from IPython.display import clear_output

# env = gym.make("FrozenLake-v0")
# register a non-slippery frozen lake
from gym.envs.registration import register


try:
    register(
            id="FrozenLakeNoSlip-v0",
            entry_point="gym.envs.toy_text:FrozenLakeEnv",
            kwargs={"map_name": "4x4", "is_slippery": False},
            max_episode_steps=100,
            reward_threshold=0.78, # optimum = .8196
            )
except:
    pass

env = gym.make("FrozenLakeNoSlip-v0")
# NOTE: default ice is slippery:
# def __init__(self, desc=None, map_name="4x4",is_slippery=True):

action_space_size = env.action_space.n
state_space_size = env.observation_space.n
#LEFT = 0 DOWN = 1 RIGHT = 2 UP = 3

q_table = np.zeros((state_space_size, action_space_size))

print(q_table)

# @hyperparameters
num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

# Exploration parameters
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

# Q-learning algorithm
for episode in range(num_episodes):
    # initialize new episode params
    state = env.reset()
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        print("q_table: ")
        print(np.around(q_table, decimals=3))
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        ## If this number > greater than epsilon --> exploitation 
        #(taking the biggest Q value for this state)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:]) 
        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)
        # print(f"new_state: {new_state}, reward: {reward}, done: {done}, info: {info}")

        # Update Q-table for Q(s,a)
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        # Set new state
        state = new_state
        # Add new reward
        rewards_current_episode += reward
        # env.render() to display the graphics
        print("Episode {}, total reward {}, epsilon: {}".format(episode, sum(rewards_all_episodes), exploration_rate))
        env.render()
        print()
        # pause to see the updates
        time.sleep(0.01)
        clear_output(wait=True)
        # check if the last action ended the episode
        if done == True: 
            break

    # Exploration rate decay
    # Reduce epsilon (because we need less and less exploration)
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

    # Add current episode reward to total rewards list
    rewards_all_episodes.append(rewards_current_episode)

    # move on to the next episode

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

# Print updated Q-table
print("\n\n********Q-table********\n")
print(q_table)

# Watch our agent play Frozen Lake by playing the best action 
# from each state according to the Q-table

# the (Left) or (Down) is indicating the Last action
# if self.lastaction is not None:
#   outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))

for episode in range(3):
    # initialize new episode params
    state = env.reset()
    done = False
    print("\n*****EPISODE ", episode+1, "*****\n")
    # time.sleep(1)

    for step in range(max_steps_per_episode):
        # clear_output(wait=True)
        # Show current state of environment on screen
        env.render()
        # time.sleep(0.3) 

        # Choose action with highest Q-value for current state
        action = np.argmax(q_table[state,:])
        # Take new action
        new_state, reward, done, info = env.step(action)

        if done:
            # clear_output(wait=True)
            env.render()
            if reward == 1:
                # Agent reached the goal and won episode
                print("****You reached the goal!****")
                # time.sleep(3)
            else:
                # Agent stepped in a hole and lost episode
                print("****You fell through a hole!****")
                # time.sleep(3)
            # clear_output(wait=True)
            break
            
        # if not done, set new state
        state = new_state
        
env.close()