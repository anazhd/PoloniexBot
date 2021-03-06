from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import *
from poloniex_env import PoloniexEnv
from agent import Agent

hidden_size = 10
nb_frames = 2
grid_w = 1
grid_h = 8

model = Sequential()
model.add(Flatten(input_shape=(nb_frames, grid_w, grid_h)))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(2))
model.compile(sgd(lr=.2), "mse")

poloniex_env = PoloniexEnv()
agent = Agent(model=model)
agent.train(poloniex_env, batch_size=10, nb_epoch=1000, epsilon = .5)
agent.play(poloniex_env)

