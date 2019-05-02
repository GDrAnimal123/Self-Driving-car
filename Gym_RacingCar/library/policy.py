import numpy as np
import random


class Policy(object):
    """Abstract base class for all implemented policies.
    Each policy helps with selection of action to take on an environment.
    Do not use this abstract base class directly but instead use one of the concrete policies implemented.
    To implement your own policy, you have to implement the following methods:
    - `select_action`
    # Arguments
        agent (rl.core.Agent): Agent used
    """

    def _set_agent(self, agent):
        self.agent = agent

    def select_action(self, **kwargs):
        raise NotImplementedError()


class LinearAnnealedPolicy(Policy):
    """Implement the linear annealing policy (epsilon decay)
    Linear Annealing Policy computes a current threshold value and
    transfers it to an inner policy which chooses the action. The threshold
    value is following a linear function decreasing over time."""

    def __init__(self, inner_policy, attr, value_max, value_min, value_test, nb_steps):
        if not hasattr(inner_policy, attr):
            raise ValueError('Policy does not have attribute "{}".'.format(attr))

        """
        # Arguments
            :param inner_policy:
                (Policy): Specify how we select optimal policy.
            :param attr:
                (String): Name of the attribute in inner_policy.
            :param epsilon:
                (FLoat): The size of the replay memory.
            :param value_max:
                (FLoat): Maximum value of the epsilon.
            :param value_min:
                (FLoat): Minimum value of the epsilon.
            :param value_test:
                (FLoat): Epsilon value during testing.
            :param nb_steps:
                (Int): Rate to decrese our epsilon linearly over time.
        """

        super(LinearAnnealedPolicy, self).__init__()

        self.inner_policy = inner_policy
        self.attr = attr
        self.value_max = value_max
        self.value_min = value_min
        self.value_test = value_test
        self.nb_steps = nb_steps

        self.epsilon = value_max

    def get_current_value(self):
        """Return current annealing value
        # Returns
            Value to use in annealing
        """
        value = self.epsilon

        if self.agent.training:
            if self.epsilon > self.value_min:
                # Linear annealed: f(x) = ax + b.
                self.epsilon -= (self.epsilon - self.value_min) / self.nb_steps
                value = self.epsilon

        else:
            value = self.value_test
        return value

    def select_action(self, **kwargs):
        """Choose an action to perform
        # Returns
            Action index to take (int)
        """
        # **kwargs)
        setattr(self.inner_policy, self.attr, self.get_current_value())
        return self.inner_policy.select_action(**kwargs)


class EpsGreedyQPolicy(Policy):
    """Implement the epsilon greedy policy
    Eps Greedy policy either:
    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """

    def __init__(self, epsilon=.1):
        super(EpsGreedyQPolicy, self).__init__()
        self.epsilon = epsilon

    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        action_size = q_values.shape[0]

        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, action_size)
        else:
            action = np.argmax(q_values)
        return action


class GreedyQPolicy(Policy):
    """Implement the greedy policy
    Greedy policy returns the current best action according to q_values
    """

    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        action = np.argmax(q_values)
        return action


class BoltzmannQPolicy(Policy):
    """Implement the Boltzmann Q Policy
    Boltzmann Q Policy builds a probability law on q values and returns
    an action selected randomly according to this law.
    """

    def __init__(self, tau=1., clip=(-500., 500.)):
        super(BoltzmannQPolicy, self).__init__()
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        return action
