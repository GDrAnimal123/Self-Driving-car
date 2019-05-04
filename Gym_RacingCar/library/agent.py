import random
import numpy as np
from collections import deque

from library.policy import GreedyQPolicy, EpsGreedyQPolicy
from utils import save_plot, normalize, standardize


class A2CAgent():
    # All the mechanic of DQN (separated from Keras model)
    def __init__(self, actor, critic, state_size, action_size,
                 value_size, observe=0, batch_size=64, gamma=.99):
        """
        # Arguments
            :param actor:
                (Keras.Model): Model for our actor (Refer to network.py in library).
            :param critic:
                (Keras.Model): Model for our critic (Refer to network.py in library).
            :param state_size:
                (Tuple): Shape of the input state.
            :param action_size:
                (Int): Number of actions that our agent can select.
            :param value_size:
                (Int): Output size of our critic model.
            :param batch_size:
                (Int): The size of batch to feed into our model.
            :param gamma:
                (Float): Discounter factor
        """

        self.state_size = state_size
        self.action_size = action_size
        self.value_size = value_size

        # These are hyper parameters for the Policy Gradient
        self.gamma = 0.99

        # Model for policy and critic network
        self.actor = actor
        self.critic = critic

        # Note: We don't need is_terminated since we make an
        # update at each step (TD learning)
        # lists for states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []
        self.batch_size = batch_size
        self.observe = observe

        # Performance Statistics
        self.stats_window_size = 50  # window size for computing rolling statistics
        self.mavg_score = []  # Moving Average of Reward
        self.var_score = []  # Variance of Reward
        self.mavg_best_value = []  # Moving Average of Value produced by Critic
        self.mavg_critic_loss = []  # Moving Average of Critic loss
        self.mavg_actor_loss = []  # Moving Average of Actor loss

    # Instead of using our agent to predict value(future reward)
    # of the next state which can produce noises or inaccuracies
    # since our agent is still training (sub-optimal).
    # Use TD(1) i.e. Monte Carlo updates
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            done = (rewards[t] == 0)
            if done:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def fit(self):
        episode_length = len(self.states)

        # These targets are used for optimization step.
        discounted_rewards = self.discount_rewards(self.rewards)
        # Standardized discounted rewards
        discounted_rewards = standardize(discounted_rewards)
        advantages = np.zeros((episode_length, self.action_size))

        # Create inputs for our model (not crucial but it helps
        # to keep track of input dimension)
        update_input = np.zeros(((episode_length,) + self.state_size))

        for i in range(episode_length):
            update_input[i, :] = self.states[i]

        # We predict on batch using list of states
        values = self.critic.predict(update_input)

        for i in range(episode_length):
            advantages[i][self.actions[i]] = discounted_rewards[i] - values[i]

        # Refer to "https://medium.freecodecamp.org/an-intro-to-advantage-actor-critic-methods-lets-play-sonic-the-hedgehog-86d6240171d"

        # Actor use Cross-entropy with critic q value
        actor_loss = self.actor.fit(update_input, advantages,
                                    batch_size=self.batch_size, epochs=1, verbose=0)
        # Critic use MSE its predicted value (value)
        critic_loss = self.critic.fit(update_input, discounted_rewards,
                                      batch_size=self.batch_size, epochs=1, verbose=0)

        self.states, self.actions, self.rewards = [], [], []

        return values, actor_loss.history['loss'], critic_loss.history['loss']

    # using the output of policy network, pick action stochastically (Stochastic Policy)
    def act(self, state):
        # policy = [0.499, 0.501]
        policy = self.actor.predict(state).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def append_memory(self, state, action, reward):
        """
        Add an observed state from the game-environment, along with the
        estimated Q-values, action taken, observed reward, etc.

        :param state:
            Current state of the game-environment.
            This is the output of the MotionTracer-class.
        :param action:
            The action taken by the agent in this state of the game.
        :param reward:
            The reward that was observed from taking this action
            and moving to the next state.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def save_weights(self, name):
        self.actor.save_weights(name + "_actor.h5", overwrite=True)
        self.critic.save_weights(name + "_critic.h5", overwrite=True)

    def load_weights(self, name):
        self.actor.load_weights(name + "_actor.h5", overwrite=True)
        self.critic.load_weights(name + "_critic.h5", overwrite=True)


class DQNAgent():
    # All the mechanic of DQN (separated from Keras model)
    def __init__(self, model, state_size, action_size, memory_size,
                 batch_size=64, policy=None, test_policy=None,
                 gamma=.99):
        """
        # Arguments
            :param model:
                (Keras.Model): Model for our dqn (Refer to network.py in library).
            :param state_size:
                (Tuple): Shape of the input state.
            :param action_size:
                (Int): Number of actions that our agent can select.
            :param memory_size:
                (Int): The size of the replay memory.
            :param batch_size:
                (Int): The size of batch to feed into our model.
            :param policy:
                (Policy): Specify how we select optimal policy during training
            :param test_policy:
                (Policy): Specify how we select optimal policy during testing
            :param gamma:
                (Float): Discounter factor
            :param training:
                (Bool): Whether for training or testing
            :param target_model_update:
                (int): How frequent we update our target model.
        """

        # # Soft vs hard target model updates.
        # if target_model_update < 0:
        #     raise ValueError('`target_model_update` must be >= 0.')
        # elif target_model_update >= 1:
        #     # Hard update every `target_model_update` steps.
        #     target_model_update = int(target_model_update)
        # else:
        #     # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
        #     target_model_update = float(target_model_update)

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma    # discount rate
        self.memory = deque(maxlen=memory_size)  # the size of replay memory
        # self.policy = policy  # policy selection
        self.training = training
        self.batch_size = batch_size

        # # Epsilon
        # self.epsilon = epsilon
        # self.initial_epsilon = 1.0
        # self.final_epsilon = 0.001
        # self.nb_steps = nb_steps

        # create main model and target model
        self.model = model
        self.target_model = None

        if policy is None:
            policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy

        self.step = 0

    def fit(self):

        num_samples = min(self.batch_size, len(self.memory))
        replay_samples = random.sample(self.memory, num_samples)

        update_input = np.zeros(((num_samples,) + self.state_size))
        update_target = np.zeros(((num_samples,) + self.state_size))

        action, reward, done = [], [], []

        i = 0
        for s_t, action_idx, r_t, s_t1, is_terminated in replay_samples:
            # Create experiences
            update_input[i, :] = s_t
            action.append(action_idx)
            reward.append(r_t)
            update_target[i, :] = s_t1
            done.append(is_terminated)
            i += 1

        target = self.model.predict(update_input)
        target_val = self.model.predict(update_target)

        i = 0
        for s_t, action_idx, r_t, s_t1, is_terminated in replay_samples:
            if is_terminated:
                target[i][action_idx] = r_t
            else:
                opt_policy = np.argmax(target_val[i])
                target[i][action_idx] = r_t + self.gamma * target_val[i][opt_policy]
            i += 1

        loss = self.model.fit(update_input, target, batch_size=self.batch_size,
                              epochs=1, verbose=0)
        return np.max(target[-1]), loss

    def compile(self, optimizer, loss='mse', metrics=[]):
        # self.target_model = clone_model(self.model, self.custom_model_objects)
        # self.target_model.compile(optimizer=optimizer, loss=loss)
        self.model.compile(loss=loss,
                           optimizer=optimizer)

    def load_weights(self, name):
        self.model.load_weights(name)

    def save_weights(self, name):
        self.model.save_weights(name)

    def act(self, state):
        # Return the optimal policy depends on our policy
        q_values = self.compute_q_values(state)
        return self.policy.select_action(q_values=q_values)

    # def act(self, state):
    #     if np.random.rand() <= self.epsilon:
    #         return random.randrange(self.action_size)
    #     q_values = self.compute_q_values(state)
    #     return np.argmax(q_values)  # returns action

    def compute_q_values(self, state):
        # Use model to predict q_values
        q_values = self.model.predict(state)[0]

        assert q_values.shape == (self.action_size,)
        return q_values

    def append_memory(self, state, action, reward, next_state, done):
        """
        Add an observed state from the game-environment, along with the
        estimated Q-values, action taken, observed reward, etc.

        :param state:
            Current state of the game-environment.
            This is the output of the MotionTracer-class.
        :param action:
            The action taken by the agent in this state of the game.
        :param reward:
            The reward that was observed from taking this action
            and moving to the next state.
        :param next_state:
            The next state of the game-environment.
        :param done:
            Boolean whether this is the end of the episode.
        """
        self.memory.append((state, action, reward, next_state, done))

        # if self.epsilon > self.final_epsilon:
        #     self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.nb_steps

    # Basically, this is how we set private variables in Python
    # Refer: https://www.python-course.eu/python3_properties.php
    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)
