from memory import ExperienceReplay
import numpy as np
import os


class BuySellAgent:
	def __init__(self, model_buy,  model_sell, memory_size=1000, nb_frames=None):

		self.model_buy = model
		self.model_sell = model
		self.memory_buy = ExperienceReplay(memory_size)
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
			E.reset()
			trade_over = False
			self.clear_frames()
			while not trade_over
				S_buy = self.getFrames( env )
				A_buy = self.getAction( S_buy, model_buy, epsilon_buy )
				E.play(A_buy)
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






		# if type(epsilon)  in {tuple, list}:
		# 	delta =  ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))
		# 	final_epsilon = epsilon[1]
		# 	epsilon = epsilon[0]
		# else:
		# 	final_epsilon = epsilon

		# win_count = 0
		# for trade in range(nb_trades):
		# 	env.reset()
		# 	frames = None
		# 	S_buy = self.get_env_data(env)
		# 	env_over = env.is_over()
		# 	while not env_over

		# 		r = 0

		# 		## Buy Phase
		# 		if np.random.random() < epsilon:
		# 			a_buy = int( np.random.randint( env.nb_actions() ) )
		# 		else:
		# 			q_buy = model_buy.predict(S_buy)
		# 			a_buy = int(np.argmax(q_buy[0]))

		# 		env.play(a_buy)
		# 		if( env.holding ): # Did buy

		# 			## Sell Phase
		# 			S_sell = self.get_env_data(env)
		# 			while env.holding():

		# 				if np.random.random() < epsilon:
		# 					a_sell = int(np.random.randint(env.nb_actions()))
		# 				else:
		# 					q_sell = model_sell.predict(S_sell)
		# 					a_sell = int(np.argmax(q_sell[0]))

		# 				r_sell = env.play(a_sell)
		# 				env_over = env.is_over()

		# 				S_sell_prime = self.get_env_data(env)
		# 				S_buy_prime = S_sell_prime
		# 				env_over = env.is_over()

		# 				transition_sell = [S_sell, a, r, S_sell_prime, env_over]
		# 				self.memory_sell.remember(*transition)
		# 				S = S_prime

		# 				batch_sell = self.memory_sell.get_batch(model=model, batch_size=batch_size, gamma=gamma)
				
		# 				if batch_sell:
		# 					inputs, targets = batch_sell
		# 					loss += float(model_sell.train_on_batch(inputs, targets))

		# 				if env_over:
		# 					transition_buy = [S_buy, a_buy, r_sell, S_buy_prime, env_over]
		# 					self.memory_buy.remember(*transition)
		# 					batch_sell = self.memory_sell.get_batch(model=model, batch_size=batch_size, gamma=gamma)
		# 					if batch_sell:
		# 						inputs, targets = batch_sell
		# 						loss += float(model_sell.train_on_batch(inputs, targets))

		# 		else:
		# 			env_over = False
		# 			S_buy_prime = self.get_env_data(env)
		# 			transition_buy = [S_buy, a, r, S_buy_prime, env_over]
		# 			self.memory_sell.remember(*transition)
		# 			S = S_prime
		# 			batch_sell = self.memory_sell.get_batch(model=model, batch_size=batch_size, gamma=gamma)
				
		# 			if batch_sell:
		# 				inputs, targets = batch_sell
		# 				loss += float(model_sell.train_on_batch(inputs, targets))



















		
