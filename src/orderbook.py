import time
import json
import heapq
import numpy as np
import matplotlib.pyplot as plt
from threading import Lock
from poloniex import PoloniexClient 
from multiprocessing.dummy import Process as Thread

class OrderBook(object):

	def __init__(self):
		self._poloniex = PoloniexClient("","")
		self._consumerThread = None
		self._symbols = []

		self._bids = {}
		self._bLock = Lock()
		self._asks = {}
		self._aLock = Lock()

	def peekBids(self, symbol):
		if self._bids[ symbol ]:
			ret = self._bids[ symbol ][0]
			ret[0] = ret[0] * -1
			return ret
		else:
			return [None, None]

	def peekAsks(self, symbol):
		asks = self._asks[ symbol ][:]
		if asks:
			return asks[0]
		else:
			return [None, None]

	def getBids(self, symbol):
		self._bLock.acquire()
		ret = self._bids[ symbol ][:]
		for bid in ret:
			bid[0] = bid[0] * -1
		return ret

	def getAsks(self, symbol):
		return self._asks[ symbol ][:]

	def addAll(self):
		ticker = self._poloniex.returnTicker()
		for sym in ticker.keys():
			self.add(sym)

	def removeAll(self):
		self._symbols = []

		self._bids = {}
		self._asks = {}

		self._consumerT.join()
		print('BOOK: consumer thread stopped')


	def add(self, symbol):
		self._symbols.append(symbol)

		self._asks[symbol] = []
		self._bids[symbol] = []

		if len(self._symbols) == 1:
			self._consumerThread = Thread(target=self.orderConsumer); 
			self._consumerThread.daemon = True
			self._consumerThread.start()
			print('BOOK: consumer thread started')

	def remove(self, symbol):

		self._symboles.remove(symbol)

		if len(self._symbols) == 0:
			self._consumerT.join()
			print('BOOK: consumer thread stopped')

	def orderConsumer(self):

		while len( self._symbols ) > 0:

			for sym in self._symbols:
				# print sym
				book = self._poloniex.returnOrderBook(sym)

				if "bids" in book :
					bids = []
					for bid in book["bids"]:

						bid[0] = -1*float(bid[0])
						bid[1] = float(bid[1])
						heapq.heappush( bids, bid )

					self._bids[sym] = bids

				if "asks" in book :
					asks = []
					for ask in book["asks"]:
						ask[0] = float(ask[0])
						ask[1] = float(ask[1])
						heapq.heappush( asks, ask )
					self._asks[sym] = asks




if __name__ == "__main__":
	book = OrderBook()
	book.add("BTC_ETH")
	# book.add("ETH_ETC")
	# book.add("BTC_ETC")

	f, = plt.plot([],[])
	plt.show(block=False)
	plt.pause(0.01)

	axes = plt.gca()


	while True:

		time.sleep(1)
		plt.pause(0.01)

		b = np.array(book.getBids("BTC_ETH"))
		b[:,1] = np.cumsum(b[:,1])

		a = np.array(book.getAsks("BTC_ETH"))
		a[:,1] = np.cumsum( a[:,1] )
		a = np.flipud(a)
		xy = np.concatenate( (a,b), axis = 0)


		y = xy[:,1]
		x = xy[:,0]

		f.set_xdata(x)
		f.set_ydata(y)

		axes.set_xlim([min(x),max(x)])
		axes.set_ylim([min(y),max(y)])

		plt.draw()




	# while True:
	# 	time.sleep(1)
	# 	btc_eth = book.peekBids("BTC_ETH")[0]
	# 	eth_etc = book.peekBids("ETH_ETC")[0]
	# 	btc_etc = book.peekAsks("BTC_ETC")[0]
	# 	print "BTC_ETH:", btc_eth
	# 	print "ETH_ETC:", eth_etc
	# 	print "BTC_ETC:", btc_etc
	# 	btc1 = 1
	# 	eth = btc1 * ( 1 / (btc_eth) ) * 0.9975
	# 	etc = eth * ( 1 / (eth_etc) ) * 0.9975
	# 	btc2 = etc * btc_etc * 0.9975
	# 	r = btc2/btc1
	# 	print "1 BTC -> {} ETH -> {} ETC -> {} BTC:  {}/{} = {}".format(eth,etc,btc2,btc1,btc2,r)	
















