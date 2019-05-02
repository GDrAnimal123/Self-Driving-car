import sys
import random
import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.optimizers import Adam

from library.network import Network
from library.agent import A2CAgent
from utils import save_plot
import CarRacing_env as environment
from CarRacing_env import render, encode_action, decode_action

env_name = environment.env_name
EPISODES = 500
STEPS = 500
TEST_INTERVAL = 10

episodes, scores, best_values, losses, actor_losses, critic_losses = [], [], [], [], [], []


if __name__ == "__main__":
    env = environment.env

    state_size = environment.state_size
    action_size = environment.action_size
    value_size = 1

    # Variables for A2C model
    batch_size = 64
    actor_lr = 0.0001
    critic_lr = 0.0001
    t = 0

    actor_model = Network.actor_network(state_size, action_size, actor_lr)
    critic_model = Network.critic_network(state_size, value_size, critic_lr)

    agent = A2CAgent(actor_model, critic_model, state_size, action_size, value_size,
                     observe=0, batch_size=batch_size, gamma=.99)

    agent.load_weights(actor_path="savepoint/CarRacing-v0-a2c-400eps-actor.h5",
                       critic_path="savepoint/CarRacing-v0-a2c-400eps-critic.h5")

    for e, _ in enumerate(range(EPISODES), start=0):
        if e % 100 == 0:
            # We should save the video of our AI playing the game
            print("Rendering....")
            render(env, agent)
            env = gym.make(env_name)

        state = env.reset()
        total_reward = 0

        # We need to reshape state for consistency (avoid extra dims)
        # We can use squeeze() from numpy but reshape gives us more control
        # on the dimension.
        # Note: state_size is a tuple of shape_dims. This is useful if our
        # input state is an pixel image which has 2 or more dims.
        state = np.expand_dims(state, axis=0)
        # for step in range(STEPS):
        for step, _ in enumerate(range(STEPS), start=1):

            # Compute the optimal action given state using model
            # Note: Epsilon is being linearly reduced when select_action
            # is called by our policy
            action_idx = agent.act(state)
            # We need to decode our action index to correspond with Racing-car env.
            action = decode_action(action_idx)

            # We feed our action to retrieve our next state, reward.
            next_state, reward, done, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)

            agent.append_memory(state, action_idx, reward)
            state = next_state
            total_reward += reward
            t += 1

            if done or (step % 200 == 0 and t > agent.observe):
                # Every episode, agent learns from sample returns
                best_value, actor_loss, critic_loss = agent.fit()

            if (step % 200 == 0 or done):
                # every episode, plot the play time
                episodes.append(e)

                scores.append(total_reward)
                best_values.append(best_value)
                actor_losses.append(actor_loss.history['loss'][0])
                critic_losses.append(critic_loss.history['loss'][0])

                save_plot(episodes, scores, './save_graph/reward_by_ep.png', "Episodes", "Scores")
                save_plot(episodes, best_values, './save_graph/best_value_by_ep.png', "Episodes", "Best Value")
                save_plot(episodes, actor_losses, './save_graph/actor_loss_by_ep.png', "Episodes", "Actor Loss")
                save_plot(episodes, critic_losses, './save_graph/critic_loss_by_ep.png', "Episodes", "Critic Loss")

                print("Episode: {0}/{1}, score: {2}, value: {3}"
                      .format(e, EPISODES, total_reward, best_value))

                # # if the mean of scores of last 10 episode is bigger than 490
                # # stop training
                # if np.mean(scores[-min(10, len(scores)):]) > 250:
                #     sys.exit()

                break

        if e % 200 == 0:
            print("Saving model at episode {}".format(e))
            # agent.save_weights(actor_path="savepoint/{}-a2c-{}eps-actor.h5".format(env_name, e),
            #                    critic_path="savepoint/{}-a2c-{}eps-critic.h5".format(env_name, e))
