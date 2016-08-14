from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
from twisted.internet.defer import inlineCallbacks
import time
import poloniex

class OrderBook(object):
	def __init__(self):
		order
	def startBook(self, symbol):
		# Starts the catcher thread
		pass
	def stopBook(self, symbol):
		# Ends the catcher thread
		pass
	def orderCatcher(self): 
		# catches orders from ordercatcher
		pass



if __name__ == "__main__":
	book = OrderBook()
	book.startBook("BTC_ETH")
	time.sleep(10)
	book.stopBook("BTC_ETH")
	
