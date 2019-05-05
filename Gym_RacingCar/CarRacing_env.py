import numpy as np
import gym
import time
from gym.wrappers.monitor import Monitor

env_name = "CarRacing-v0"
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

# register(
#     id='CarRacing-v0',
#     entry_point='gym.envs.box2d:CarRacing',
#     max_episode_steps=1000,
#     reward_threshold=900,
# )
env = gym.make(env_name)

# State is a pixel image of (96, 96, 3)
state_size = env.observation_space.shape
action_size = len([UP, DOWN, LEFT, RIGHT])
STEPS = 500


def encode_action(ctn_action):
    '''
    # Arguments
        :param ctn_action(np.array) : [-1.0|+1.0, 0.0|1.0, 0.0|0.8]

    Return: action_index(Int) : UP|DOWN|LEFT|RIGHT

    '''
    # Encode the continuous action from game environment
    # to onehot action and feed them to our Keras Model.

    action_index = 0

    if ctn_action[0] < 0:
        action_index = LEFT
    if ctn_action[0] > 0:
        action_index = RIGHT
    if ctn_action[1] > 0:
        action_index = UP
    if ctn_action[2] > 0:
        action_index = DOWN

    return action_index


def decode_action(action_index):
    '''
    # Arguments
        :param action(Int): [UP, DOWN, LEFT, RIGHT]

    Return: ctn_action(np.array): [-1.0|+1.0, 0.0|1.0, 0.0|0.8]
    '''
    # Decode the action that our Keras Model predict
    # to continuous action that used in game environment
    a = np.zeros(3)

    if action_index == LEFT:
        a[0] = -1.0
    if action_index == RIGHT:
        a[0] = +1.0
    if action_index == UP:
        a[1] = +1.0
    if action_index == DOWN:
        a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    return a


def render(env, agent, name="", record=False):
    if record:
        env = Monitor(env, './video-test/{}'.format(name), force=True, mode="evaluation")
    for i_episode in range(5):
        state = env.reset()
        total_reward = 0
        for step, _ in enumerate(range(STEPS), start=1):
            state = np.expand_dims(state, axis=0)
            env.render()

            action_index = agent.act(state)
            action = decode_action(action_index)

            next_state, reward, done, info = env.step(action)
            if done:
                break
            state = next_state
            total_reward += reward

        print("Episode achieves total reward {}".format(total_reward))

    # env.close()
