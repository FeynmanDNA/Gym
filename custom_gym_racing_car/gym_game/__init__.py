# init file for the custom environment

from gym.envs.registration import register


register(
    # id for main program to make the env
    id='Pygame-v0',
    # entry point is the env Class
    entry_point='gym_game.envs:CustomEnv',
    max_episode_steps=2000,
)

