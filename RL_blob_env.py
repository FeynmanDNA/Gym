import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time


style.use("ggplot")

# CONSTANTS

# no observation spaces, just grid as env
SIZE = 10
# somewhere between 25k and 75K
# training episodes were required for the 10x10 to learn
HM_EPISODES = 25000  # how many episodes
# for 20x20 model will take 2500k episodes
max_steps_per_episode = 200
MOVE_PENALTY = 1
ENEMY_PENALTY = 300  # if run into enemy
FOOD_REWARD = 25


# exploration parameters
epsilon = 0.9
EPS_DECAY = 0.9998

# how often to display
SHOW_EVERY = 3000

start_q_table = None  # or filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

# key in dictionary
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

d = {
     1: (255, 175, 0), # player blue
     2: (0, 200, 0),  # food green
     3: (0,0,255) # enemy red
    }

# class for Blob, with attributes and methods
class Blob:
    def __init__(self):
        # starting position xy
        # 0-9
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
    def __str__(self):
        return f"({self.x}, {self.y})"
    def __sub__(self, other):
        # substract a blob from another blob
        return (self.x - other.x, self.y - other.y)
    def action(self, choice):
        # action space
        # can only move diagonally
        if choice == 0:
            # move top right
            self.move(x=1, y=1)
        elif choice == 1:
            # move bottom left
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
    def move(self, x=False, y=False):
        # move randomly when no specific x y is passed
        if not x:
            # NOTE: random move can be not just diagonally
            self.x += np.random.randint(-1, 2)  # -1,0,1
        else:
            self.x += x
        self.x = max(0, self.x)
        self.x = min(self.x, SIZE-1)
        if not y:
            # NOTE: random move can be not just diagonally
            self.y += np.random.randint(-1, 2)  # -1,0,1
        else:
            self.y += y
        self.y = max(0, self.y)
        self.y = min(self.y, SIZE-1)

# init q_table
if start_q_table is None:
    # create new q table
    q_table = {}
    # observation will be relative distance
    # from player to Food (x1, y1)
    # from player to enemy (x2, y2)
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    # the key to the q_table is
                    # a tuple of tuples
                    q_table[( (x1, y1), (x2, y2) )] = [
                            # four random vales
                            # we have 4 discrete actions
                            np.random.uniform(-5,0)
                            for i in range(4)
                    ]
                    """
                    {((-9, -9), (-9, -9)): [-1.1562249063525205,
                      -3.6334160754756377,
                      -0.12277002460631259,
                      -4.485412008694313],
                     ((-9, -9), (-9, -8)): [-3.9828080378992152,
                      -0.14882594477723998,
                      -2.5752050379787383,
                      -4.743879660914775],
                      ...
                    """
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


################# main RL program ##################

episode_rewards = []

for episode in range(HM_EPISODES):
    # reset state
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY == 0:
        print(f"on Episode {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} episode mean reward = {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False


    episode_reward = 0
    for i in range(max_steps_per_episode):
        observation = (player-food, player-enemy)
        # Exploration-exploitation trade-off
        exploration_rate_threshold = np.random.random()
        ## If this number > greater than epsilon --> exploitation 
        #(taking the biggest Q value for this state)
        if exploration_rate_threshold > epsilon:
            action = np.argmax(q_table[observation])
        # Else doing a random choice --> exploration
        else:
            action = np.random.randint(0,4)

        player.action(action)

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        # calculate reward
        episode_reward += reward

        # new observation
        new_observation = (player-food, player-enemy)

        # calculate q
        max_future_q = np.max(q_table[new_observation])
        current_q = q_table[observation][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q \
                    + LEARNING_RATE * \
                    (reward + DISCOUNT * max_future_q)

        # update q_table
        q_table[observation][action]  = new_q

        if show:
            time.sleep(0.1)
            env = np.zeros((SIZE,SIZE,3), dtype=np.uint8)
            env[food.y][food.x] = d[FOOD_N]  # get the color
            env[player.y][player.x] = d[PLAYER_N]  # get the color
            env[enemy.y][enemy.x] = d[ENEMY_N]  # get the color

            # show img from array
            img = Image.fromarray(env, "RGB")
            img = img.resize((300,300))
            cv2.imshow("RL Blob env", np.array(img))

            # episode end and wait
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # terminate
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    episode_rewards.append(episode_reward)

    # decay epsilon
    epsilon *= EPS_DECAY


#### graph the stats
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward per {SHOW_EVERY}")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-RL-blob-{HM_EPISODES}.pickle", "wb") as f:
    pickle.dump(q_table, f)

