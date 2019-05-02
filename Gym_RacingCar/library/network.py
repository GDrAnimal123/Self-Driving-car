from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.optimizers import Adam


class Network(object):
    @staticmethod
    def dqn(state_size, action_size, learning_rate):

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=state_size))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(action_size, activation='linear'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model

    @staticmethod
    def actor_network(state_size, action_size, learning_rate):
        """Actor Network for A2C
        """

        model = Sequential()
        model.add(Conv2D(32, (8, 8), input_shape=(state_size)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(64, (4, 4)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(action_size, activation='softmax'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=adam)
        model.summary()

        return model

    @staticmethod
    def critic_network(state_size, value_size, learning_rate):
        """Critic Network for A2C
        """

        model = Sequential()
        model.add(Conv2D(32, (8, 8), input_shape=(state_size)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(64, (4, 4)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(value_size, activation='linear'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=adam)
        model.summary()

        return model

    @staticmethod
    def actor_network_cartpole(state_size, action_size, learning_rate):
        model = Sequential()
        model.add(Dense(24, input_shape=state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=learning_rate))
        return model

    @staticmethod
    def critic_network_cartpole(state_size, value_size, learning_rate):
        model = Sequential()
        model.add(Dense(24, input_shape=state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(value_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
        return model


# def actor_network_cartpole(state_size, action_size, learning_rate):
#     """Actor Network for A2C
#     """

#     model = Sequential()
#     model.add(Dense(32, input_shape=state_size))
#     model.add(Activation('relu'))
#     model.add(Dense(32))
#     model.add(Activation('relu'))
#     model.add(Dense(32))
#     model.add(Activation('relu'))
#     model.add(Dense(action_size, activation='softmax'))

#     adam = Adam(lr=learning_rate)
#     model.compile(loss='categorical_crossentropy', optimizer=adam)

#     return model


# def critic_network_cartpole(state_size, value_size, learning_rate):
#     """Critic Network for A2C
#     """

#     model = Sequential()
#     model.add(Dense(32, input_shape=state_size))
#     model.add(Activation('relu'))
#     model.add(Dense(32))
#     model.add(Activation('relu'))
#     model.add(Dense(32))
#     model.add(Activation('relu'))

#     model.add(Dense(value_size, activation='linear'))

#     adam = Adam(lr=learning_rate)
#     model.compile(loss='mse', optimizer=adam)

#     return model
