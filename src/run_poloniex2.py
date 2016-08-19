from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import *
from poloniex_env2 import PoloniexEnv2
from agent import Agent

hidden_size = 10
nb_frames = 10
grid_w = 1
grid_h = 8

model = Sequential()
model.add(Flatten(input_shape=(nb_frames, grid_w, grid_h)))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(2))
model.compile(sgd(lr=.2), "mse")


poloniex_env = PoloniexEnv2()
agent = Agent(model=model)
agent.train(poloniex_env, batch_size=10, nb_epoch=1000, epsilon = .01)

# serialize model to JSON
model_json = model.to_json()
with open("models/model2.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("models/model2.h5")
print("Saved model to disk")
