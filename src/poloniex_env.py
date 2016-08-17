__author__ = "JRS"

import numpy as np
from game import Game
import time
import heapq
import poloniex

class PoloniexEnv(Game):

	def __init__(self):
		self._symbol = "BTC_ETH"
		self._poloniex = poloniex.PoloniexClient(" ", " ")
		self.reset()


	def reset(self):
		print "Reset"
		self._btc = 1
		self._eth = 0
		self._over = False


	@property
	def name(self):
		return "PoloniexEnv"

	
	def nb_actions(self):
		return 2

	def is_over(self):
		return self._over

	def is_won(self):

		if( self._over and (self._btc - 1) > 0 ):
			return True
		else:
			return False


	def play(self, action):
		
		time.sleep(1)

		reward = 0

		# Buy
		if action == 1 and self._btc == 1:
			asks,bids = self.getOrderBook()
			rate = bids[0][0]
			self._eth = self._btc * 1 / rate
			self._btc = 0
			print "buy at {} for {} ETH".format(rate, self._eth)
	
		# Sell
		elif action == 0 and self._btc == 0:
			asks,bids = self.getOrderBook()
			rate = asks[0][0]
			self._btc = self._eth * rate
			self._eth = 0
			self._over = True
			reward = self._btc - 1
			print "sell at {} for {} BTC".format(rate, self._btc)

		return reward

	def get_position(self):
		if ( self._btc == 0 ):
			asks,bids = self.getOrderBook()
			rate = asks[0][0]
			btc_temp = self._eth * rate
			reward = btc_temp - 1
			return reward

		else:
			return 0

	def get_state(self):

		b,a = self.getOrderBook()
		b = np.array(b)
		a = np.array(a)
		
		state = np.asarray([[ self._btc, self.get_position(), self.getTick(), b[:,1].mean(), b[:,0].mean(), b[:,1].sum(), a[:,0].mean(), a[:,1].mean(), a[:,1].sum() ]])
		print "S:", state[0].tolist()
		return state

	def getTick(self):
		return float(self._poloniex.returnTicker()["BTC_ETH"]["last"])


	def getOrderBook(self):
		book = self._poloniex.returnOrderBook(self._symbol)

		if "bids" in book :
			bids = []
			for bid in book["bids"]:

				bid[0] = float(bid[0])
				bid[1] = float(bid[1])
				bids.append(bid)

		if "asks" in book :
			asks = []
			for ask in book["asks"]:
				ask[0] = float(ask[0])
				ask[1] = float(ask[1])
				asks.append(ask)

		asks.sort()
		bids.sort(reverse=True)

		return bids, asks



if __name__ == "__main__":
	env = PoloniexEnv()

	print env.get_state()
	# b,a = env.getOrderBook()

	# b = np.array(b)
	# print b[:,1].mean()
	# print b[:,0].mean()
	# print b[:,1].sum()
	# print a[]
	# b[:,1] = np.cumsum(b[:,1])

	# a = np.array(a)
	# print (a[0,0] - b[0,0])
	# a[:,1] = np.cumsum( a[:,1] )
	# a = np.flipud(a)
	# xy = np.concatenate( (a,b), axis = 0)
	# print xy[x]

	

		 
