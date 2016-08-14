import time
import poloniex
from subprocess import Popen, PIPE
from multiprocessing.dummy import Process as Thread

class OrderBook(object):
	def __init__(self):
		self._receiverT = None
		self._catchersP = {}

	def startBook(self, symbol):
		if not self._catchersP:
			self._receiverT = Thread(target=self.orderReceiver); 
			self._receiverT.daemon = True
			self._receiverT.start()
                	print('BOOK: receiver thread started')

		self._catchersP[ symbol ] = Popen(["python", "ordercatcher.py", str(symbol) ], stdin=PIPE, stdout=PIPE, bufsize=1)
                print('BOOK: {} process started'.format(symbol))

	def stopBook(self, symbol):
	
		self._catchersP[ symbol ].terminate()	
		self._catchersP[ symbol ].kill()
		del self._catchersP[ symbol ]
		print ('BOOK: {} process stopped'.format(symbol))
		if not self._catchersP:
			self._receiverT.join()
			print('BOOK: reciever thread stopped')
			
	def orderReceiver(self): 
		# catches orders from ordercatcher
		pass



if __name__ == "__main__":
	book = OrderBook()
	book.startBook("BTC_ETH")
	time.sleep(2)
	book.stopBook("BTC_ETH")
	
