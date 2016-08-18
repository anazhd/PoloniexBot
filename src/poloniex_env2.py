__author__ = "JRS"

import numpy as np
from game import Game
import time
import heapq
import poloniex
import sys


class PoloniexEnv2(Game):

	def __init__(self):
		self._symbol = "BTC_ETH"
		self._poloniex = poloniex.PoloniexClient(" ", " ")
		self.reset()

	def reset(self):
		print "Reset"
		asks, bids = self.getOrderBook()
		self._position = 0
		self._bid_last = bids[0][0]
		self._ask_last = asks[0][0]
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
		
		asks, bids = self.getOrderBook()
		bid_now = bids[0][0] # best buy price
		ask_now = asks[0][0] # best sell price

		# Buy Phase
		if self._btc == 1:

			# Buy ETH
			if action == 1: 
				self._eth = self._btc * 1 / bid_now
				self._btc = 0
				print "buy at {} for {} ETH".format(bid_now, self._eth)


			# Hold BTC
			else:  
				print "hold {} BTC with position {}".format(self._btc, self._bid_last - bid_now)

			# ( + for decrease, - for increase )
			self._position = self._bid_last - bid_now
			reward = self._position

		# Sell Phase
		elif self._btc == 0:

			# Sell ETH
			if action == 0 and self._btc == 0:
				self._btc = self._eth * ask_now
				self._eth = 0
				self._over = True
				print "sell at {} for {} BTC".format(ask_now, self._btc)


			#Hold ETH
			else:
				print "hold {} ETH with position {}".format(self._eth, ask_now - self._ask_last)

			# ( - for decrease, + for increase )
			self._position = ask_now - self._ask_last
			reward = self._position

		self._bid_last = bid_now
		self._ask_last = ask_now

		self.waitUntilNextTick()
		return reward


	def get_state(self):

		b,a = self.getOrderBook()
		b = np.array(b)
		a = np.array(a)
		
		state = np.asarray([[ self._position, self.getTick(), b[:,1].mean(), b[:,0].mean(), b[:,1].sum(), a[:,0].mean(), a[:,1].mean(), a[:,1].sum() ]])
		# print "S:", state[0].tolist()
		return state

	def getTick(self):
		return float(self._poloniex.returnTicker()[self._symbol]["last"])

	def waitUntilNextTick(self):
		current = self.getTick()
		while( current == self.getTick() ):
			time.sleep(0.25)
			sys.stdout.write('.')
			sys.stdout.flush()
		print ""


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
