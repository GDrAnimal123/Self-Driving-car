import sys
import time
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import gym
from gym.wrappers.monitor import Monitor

import tensorflow as tf
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.optimizers import Adam, RMSprop

from library.network import Network
from library.agent import A2CAgent
from library.atari_wrappers import FrameStack, WarpFrame
from utils import save_plot, normalize, standardize
import CarRacing_env as environment
from CarRacing_env import render, encode_action, decode_action

env_name = environment.env_name
EPISODES = 100_000
STEPS = 400

# total_timesteps = int(80e6)

scores, best_values, losses, actor_losses, critic_losses = [], [], [], [], []

if __name__ == "__main__":

    # Setup environment
    env = environment.env

    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    img_rows, img_cols = 64, 64
    img_channels = 4  # We stack 4 frames

    state_size = (img_rows, img_cols, img_channels)
    action_size = environment.action_size
    value_size = 1

    # Variables for A2C model
    nenvs = 1
    batch_size = nenvs * STEPS

    # lr = 7e-4  # (7 * 10^-4) = 0.0007
    lr = 7e-3
    optimizer = RMSprop(lr=lr, epsilon=1e-5, decay=0.99, clipvalue=0.5)
    actor_model = Network.actor_network(state_size, action_size, optimizer)
    critic_model = Network.critic_network(state_size, value_size, optimizer)

    agent = A2CAgent(actor_model, critic_model, state_size, action_size, value_size,
                     observe=0, batch_size=batch_size, gamma=.99)

    # Start training
    GAME = 0
    t = 0
    max_reward = 0  # Maximum episode life (Proxy for agent performance)

    """Warp frames to custom (width, height)."""
    env = WarpFrame(env, width=img_rows, height=img_cols, grayscale=True)
    """Stack k last frames"""
    env = FrameStack(env, img_channels)

    SAVED_EPISODE = 3740
    try:
        print("Loading pretrain model....")
        agent.load_weights("savepoint/CarRacing-v0-{}episode".format(SAVED_EPISODE))
    except Exception as error:
        print(error.strerror)

    for episode, _ in enumerate(range(EPISODES), start=1):

        state = env.reset()
        state = np.array(state)  # (64, 64, 4)
        # Expand 1st dimension in feed into our model
        state = np.expand_dims(state, axis=0)  # (1, 64, 64, 4)

        total_reward = 0

        done = False
        # for step, _ in enumerate(range(STEPS), start=1):
        while not done:

            loss = 0  # Training Loss at each update

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

            if episode % 200 == 0:
                print("Saving model at episode {}".format(SAVED_EPISODE + episode))
                agent.save_weights(name="savepoint/{}-{}episode".format(env_name, SAVED_EPISODE + episode))

            if (done and t > agent.observe):
                # Every episode, agent learns from sample returns
                values, actor_loss, critic_loss = agent.fit()

            if done:
                if total_reward > max_reward:
                    max_reward = total_reward

                best_value = np.max(values)
                actor_loss_value = actor_loss[0]
                critic_loss_value = critic_loss[0]

                scores.append(total_reward)
                best_values.append(best_value)
                actor_losses.append(actor_loss_value)
                critic_losses.append(critic_loss_value)

                print("Episode: {0}/{1}, score: {2}, value: {3}, actor_loss: {4}, critic_loss: {5}"
                      .format(episode, EPISODES, total_reward, best_value, actor_loss_value, critic_loss_value))

                # every episode, plot the statistic
                if episode % agent.stats_window_size == 0 and t > agent.observe:
                    print("SAVE STATISTICS...")
                    agent.mavg_score.append(np.mean(np.array(scores)))
                    agent.var_score.append(np.var(np.array(scores)))
                    agent.mavg_best_value.append(np.mean(np.array(best_values)))
                    agent.mavg_critic_loss.append(np.mean(np.array(actor_losses)))
                    agent.mavg_actor_loss.append(np.mean(np.array(critic_losses)))

                    n_steps = range(len(agent.mavg_score))
                    save_plot(n_steps, agent.mavg_score, './save_graph/avg_reward_by_ep.png', "Step", "Average Scores")
                    save_plot(n_steps, agent.var_score, './save_graph/var_reward_by_ep.png', "Step", "Variance Scores")
                    save_plot(n_steps, agent.mavg_best_value, './save_graph/best_value_by_ep.png', "Step", "Best Value")
                    save_plot(n_steps[1:], agent.mavg_critic_loss[1:], './save_graph/actor_loss_by_ep.png', "Step", "Actor Loss")
                    save_plot(n_steps[1:], agent.mavg_actor_loss[1:], './save_graph/critic_loss_by_ep.png', "Step", "Critic Loss")

                    with open("statistics/a2c_stats.txt", "w") as stats_file:
                        stats_file.write('Game: ' + str(episode) + '\n')
                        stats_file.write('Learning rate: ' + str(lr) + '\n')
                        stats_file.write('Max Score: ' + str(max_reward) + '\n')
                        stats_file.write('mavg_score: ' + str(agent.mavg_score) + '\n')
                        stats_file.write('var_score: ' + str(agent.var_score) + '\n')
                        stats_file.write('mavg_best_value: ' + str(agent.mavg_best_value) + '\n')

        # if len(scores) > 0:
        #     print("Total reward: {} and max reward: {}".format(total_reward, np.max(scores)))
        #     if total_reward >= np.max(scores):
        #         print("Best performance: {}", total_reward)
        #         # We should save the video of our AI playing the game
        #         print("Rendering....")
        #         render(env, agent, name="{}-{}".format(timestamp, episode))

    env.close()
