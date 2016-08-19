from memory import ExperienceReplay
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as img
import os

class Agent:

	def __init__(self, model, memory=None, memory_size=1000, nb_frames=None):
		assert len(model.output_shape) == 2, "Model's output shape should be (nb_samples, nb_actions)."
		if memory:
			self.memory = memory
		else:
			self.memory = ExperienceReplay(memory_size)
		if not nb_frames and not model.input_shape[1]:
			raise Exception("Missing argument : nb_frames not provided")
		elif not nb_frames:
			nb_frames = model.input_shape[1]
		elif model.input_shape[1] and nb_frames and model.input_shape[1] != nb_frames:
			raise Exception("Dimension mismatch : time dimension of model should be equal to nb_frames.")
		self.model = model
		self.nb_frames = nb_frames
		self.frames = None

	@property
	def memory_size(self):
		return self.memory.memory_size

	@memory_size.setter
	def memory_size(self, value):
		self.memory.memory_size = value

	def reset_memory(self):
		self.exp_replay.reset_memory()

	def check_env_compatibility(self, env):
		env_output_shape = (1, None) + env.get_frame().shape

		if len(env_output_shape) != len(self.model.input_shape):
			raise Exception('Dimension mismatch. Input shape of the model should be compatible with the env.')
		else:
			for i in range(len(self.model.input_shape)):
				if self.model.input_shape[i] and env_output_shape[i] and self.model.input_shape[i] != env_output_shape[i]:
					raise Exception('Dimension mismatch. Input shape of the model should be compatible with the env.')

		if len(self.model.output_shape) != 2 or self.model.output_shape[1] != env.nb_actions():
			raise Exception('Output shape of model should be (nb_samples, nb_actions).')

	def get_env_data(self, env):
		frame = env.get_frame()
		if self.frames is None:
			self.frames = [frame] * self.nb_frames
		else:
			self.frames.append(frame)
			self.frames.pop(0)
		return np.expand_dims(self.frames, 0)

	def clear_frames(self):
		self.frames = None




	def train(self, env, nb_epoch=1000, batch_size=50, gamma=0.9, epsilon=[1., .1], epsilon_rate=0.5, reset_memory=False):
		self.check_env_compatibility(env)
		if type(epsilon)  in {tuple, list}:
			delta =  ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))
			final_epsilon = epsilon[1]
			epsilon = epsilon[0]
		else:
			final_epsilon = epsilon
		model = self.model
		nb_actions = model.output_shape[-1]
		win_count = 0
		for epoch in range(nb_epoch):
			loss = 0.
			env.reset()
			self.clear_frames()
			if reset_memory:
				self.reset_memory()
			env_over = False
			S = self.get_env_data(env)
			while not env_over:

				if np.random.random() < epsilon:
					a = int(np.random.randint(env.nb_actions()))
				else:
					q = model.predict(S)
					a = int(np.argmax(q[0]))


				r = env.play(a)


				S_prime = self.get_env_data(env)
				env_over = env.is_over()
				transition = [S, a, r, S_prime, env_over]

				self.memory.remember(*transition)

				S = S_prime

				batch = self.memory.get_batch(model=model, batch_size=batch_size, gamma=gamma)
				
				if batch:
					inputs, targets = batch
					loss += float(model.train_on_batch(inputs, targets))

			if env.is_won():
				win_count += 1
			if epsilon > final_epsilon:
				epsilon -= delta
			print("Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | Win count {}".format(epoch + 1, nb_epoch, loss, epsilon, win_count))

	def play(self, env, nb_epoch=10, epsilon=0., visualize=True):
		self.check_env_compatibility(env)
		model = self.model
		win_count = 0
		frames = []
		for epoch in range(nb_epoch):
			env.reset()
			self.clear_frames()
			S = self.get_env_data(env)
			if visualize:
				frames.append(env.draw())
			env_over = False
			while not env_over:
				if np.random.rand() < epsilon:
					print("random")
					action = int(np.random.randint(0, env.nb_actions))
				else:
					q = model.predict(S)			
					action = int(np.argmax(q[0]))
				env.play(action)
				S = self.get_env_data(env)
				if visualize:
					frames.append(env.draw())
				env_over = env.is_over()
			if env.is_won():
				win_count += 1
		print("Accuracy {} %".format(100. * win_count / nb_epoch))

		if visualize:
			if 'images' not in os.listdir('.'):
				os.mkdir('images')
			for i in range(len(frames)):
				plt.imshow(frames[i], interpolation='none')
				plt.show(block=True)
				plt.pause(0.01)
				plt.savefig("images/" + env.name + str(i) + ".png")


class BuySellAgent:

	def __init__(self, model_buy,  model_sell, memory_size=1000, nb_frames=None):

		self.model_buy = model_buy
		self.memory_buy = ExperienceReplay(memory_size)
		self.model_sell = model_sell
		self.memory_sell = ExperienceReplay(memory_size)
		self.nb_frames = nb_frames
		self.frames = None

	def getAction(state, model, epsilon):
		q = model.predict(S)
		if  np.random.random() < epsilon:
			a = int(np.random.randint(len(q)))
		else:
			a = int(np.argmax(q_sell[0]))

	def teach(model, memory, batch_size, gamma):
		batch_sell = memory.get_batch(model=model, batch_size=batch_size, gamma=gamma)
		if batch_sell:
			inputs, targets = batch_sell
			loss += float(model.train_on_batch(inputs, targets))
				
	def decEpsilon(epsilon, final, delta):
		if epsilon > final:
			epsilon -= delta
		return epsilon

	def getFrame(self, env):
		frame = env.get_frame()
		if self.frames is None:
			self.frames = [frame] * self.nb_frames
		else:
			self.frames.append(frame)
			self.frames.pop(0)
		return np.expand_dims(self.frames, 0)

	def clear_frames(self):
		self.frames = None


	def train(self, env, nb_trades=1000, batch_size=50, gamma=0.9, epsilon=[1., .1], epsilon_rate=0.5):

		if type(epsilon)  in {tuple, list}:
			delta =  ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))
			final_epsilon = epsilon[1]
			epsilon = epsilon[0]
		else:
			final_epsilon = epsilon

		epsilon_buy = epsilon
		epsilon_sell = epsilon
		model_sell = self.model_sell
		model_buy = self.model_buy
		memory_sell = self.memory_sell
		memory_buy = self.memory_buy

		for trade in range(nb_trades):
			env.reset()
			trade_over = False
			self.clear_frames()
			while not trade_over:
				S_buy = self.getFrames( env )
				A_buy = self.getAction( S_buy, model_buy, epsilon_buy )
				env.play(A_buy)
				if A_buy == 1: # Buy
					S_sell = S_buy_prime
					while not trade_over:
						A_sell  = self.getAction( S_sell, model_sell, epsilon_sell ) 
						E.play(A_sell)
						S_sell_prime = self.get_env_data( env )
						if A_sell == 1:  # Sell
							R = E.getProfit()
							trade_over = True
						else:			 # Hold
							R = 0
						transition_sell = [S_sell, A_sell, R, S_sell_prime, trade_over]
						self.memory_sell.remember(*transition_sell)
						teach( model_sell, memory_sell, batch_size, gamma )
						epsilon_sell = decEpsilon( epsilon_sell )

					S_buy_prime = S_sell_prime
				else:          # Hold
					R = 0
					S_buy_prime = self.get_env_data

				transition_buy = [S_buy, A_buy, R, S_buy_prime, trade_over]
				self.memory_buy.remember(*transition_buy)
				teach( model_buy, memory_buy, batch_size, gamma)
				epsilon_buy = decEpsilon( epsilon_buy, final_epsilon, delta )

 
