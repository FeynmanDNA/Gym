import gym
import random
import numpy as np
import time
# register a non-slippery frozen lake
from gym.envs.registration import register
# ipython clear output
from IPython.display import clear_output

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


# discrete action space example with MountainCar-v0
env_name = "FrozenLakeNoSlip-v0"
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

class Q_agent(Agent):
    """
    a sub class of the random Agent()
    """
    def __init__(self, env, discount=0.99, lr=0.1, epsilon=1):
        super().__init__(env)
        self.discount = discount
        self.state_size = env.observation_space.n
        self.learning_rate = lr
        self.epsilon = epsilon
        print("Q_agent state size: ", self.state_size)
        print("Q_agent action size: ", self.action_size)
        print("Q_agent discount: ", self.discount)
        self.build_model()

    def build_model(self):
        """keep track of the q table"""
        # self.q_table = 1e-4*np.random.random((self.state_size, self.action_size))
        self.q_table = np.zeros((self.state_size, self.action_size))

    def get_action(self, state):
        """decide which action to take, based on the input state"""
        q_state = self.q_table[state]
        print("q_state: ")
        print(q_state)

        """instead of only selecting actions greedily based on the
        randomly initialized q-table,
        we make the agent explore other actions """
        action_exploit = np.argmax(q_state)
        action_explore = super().get_action(state)
        if random.random() < self.epsilon:
            action = action_explore
        else:
            action = action_exploit
        return action

    def train(self, state, action, next_state, reward, done):
        """update the q table at each step"""
        # q_next state
        if done:
            q_next = np.zeros([self.action_size])
        else:
            q_next = self.q_table[next_state]

        print("q_next state: ", next_state)
        print(q_next)

        q_target = reward + self.discount*np.max(q_next)

        print("q_target: ", q_target)

        q_update = q_target - self.q_table[state, action]

        print("q_update: ", q_update)
        print("self.learning_rate * q_update: ", self.learning_rate * q_update)

        self.q_table[state, action] += self.learning_rate * q_update

        print("updated q_table for last state: ", self.q_table[state])

        # reduce epsilon after each episode
        if done:
            self.epsilon = max(0.1, self.epsilon * 0.9999)

agent = Q_agent(env)

total_reward = 0

for episode in range(10000):
    #  print("\n***********new episode *****************\n")
    # initial state
    state = env.reset()
    done = False
    while not done:
        print("q_table:")
        print(np.around(agent.q_table, decimals=4))
        # action_space.sample() get a random action
        # action = env.action_space.sample()
        action = agent.get_action(state)
        print("last state: ", state, "action: ", action)
        # apply the action to the environment
        next_state, reward, done, info = env.step(action)
        # train the q_agent and update q table
        agent.train(state, action, next_state, reward, done)
        # keep track of the cumulative reward
        total_reward += reward
        # new state
        state = next_state
        print("Episode {}, total reward {}, epsilon: {}".format(episode, total_reward, agent.epsilon))
        # env.render() to display the graphics
        env.render()
        print()
        # pause to see the updates
        time.sleep(0.1)
        clear_output(wait=True)
