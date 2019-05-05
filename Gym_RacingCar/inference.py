from keras.optimizers import Adam, RMSprop

from library.network import Network
from library.agent import A2CAgent
from library.atari_wrappers import FrameStack, WarpFrame

import CarRacing_env as environment
from CarRacing_env import render

# Setup environment
env = environment.env

img_rows, img_cols = 64, 64
img_channels = 4  # We stack 4 frames

state_size = (img_rows, img_cols, img_channels)
action_size = environment.action_size
value_size = 1

STEPS = 200
batch_size = 1 * STEPS

lr = 7e-4  # (7 * 10^-4) = 0.0007
optimizer = RMSprop(lr=lr, epsilon=1e-5, decay=0.99, clipvalue=0.5)
actor_model = Network.actor_network(state_size, action_size, optimizer)
critic_model = Network.critic_network(state_size, value_size, optimizer)

agent = A2CAgent(actor_model, critic_model, state_size, action_size, value_size,
                 observe=0, batch_size=batch_size, gamma=.99)
try:
    agent.load_weights("savepoint/CarRacing-v0-220episode")
except Exception as error:
    print(error.strerror)

env = WarpFrame(env, width=img_rows, height=img_cols, grayscale=True)
env = FrameStack(env, img_channels)

render(env, agent, record=False)
