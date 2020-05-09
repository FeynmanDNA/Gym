import gym
import random


env_name = "CartPole-v1"
env = gym.make(env_name)

# observation space
print("Observation space: ", env.observation_space)
# action space
print("Action space: ", env.action_space)

class Agent():
    def __init__(self, env):
        self.action_space_size = env.action_space.n
        print("Action space size: ", self.action_space_size)

    def get_action(self, state):
        # action = random.choice(range(self.action_space_size))
        pole_angle = state[2]
        # discrete action space
        action = 0 if pole_angle < 0 else 1
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
