import gym
import random
import numpy as np


# discrete action space example with MountainCar-v0
env_name = "MountainCar-v0"
# continuous action space with MountainCarContinuous-v0
env_name = "MountainCarContinuous-v0"
env = gym.make(env_name)

# observation space
print("Observation space: ", env.observation_space)
# action space
print("Action space: ", env.action_space)

class Agent():
    def __init__(self, env):
        self.is_discrete = (type(env.action_space) == gym.spaces.discrete.Discrete)

        if self.is_discrete:
            # action_space.n is for discrete action space
            self.action_size = env.action_space.n
            print("Action size:", self.action_size)
        else:
            # action_space.low and action_space.high
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_shape = env.action_space.shape
            print("Action range:", self.action_low, self.action_high)

    def get_action(self, state):
        if self.is_discrete:
            # discrete action space
            #  pole_angle = state[2]
            #  action = 0 if pole_angle < 0 else 1
            action = random.choice(range(self.action_size))
        else:
            action = np.random.uniform(self.action_low,
                                       self.action_high,
                                       self.action_shape)
        return action


agent = Agent(env)
# initial state
state = env.reset()

for time_step in range(200):
    # action_space.sample() get a random action
    # action = env.action_space.sample()
    action = agent.get_action(state)
    # apply the action to the environment
    state, reward, done, info = env.step(action)
    # env.render() to display the graphics
    env.render()
