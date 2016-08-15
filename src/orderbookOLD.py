import time
import json
import poloniex
from subprocess import Popen, PIPE
from multiprocessing.dummy import Process as Thread

class OrderBook(object):
	def __init__(self):
		self._consumerT = None
		self._catchersP = {}
		self._subPrc = True

	def startBook(self, symbol):
		
		self.subPrc = True
		if not self._catchersP:
			self._consumerT = Thread(target=self.orderConsumer); 
			self._consumerT.daemon = True
			self._consumerT.start()
                	print('BOOK: consumer thread started')

		self._catchersP[ symbol ] = Popen(["python", "ordercatcher.py", str(symbol) ], stdin=PIPE, stdout=PIPE, bufsize=1)
                print('BOOK: {} process started'.format(symbol))
		
		

	def stopBook(self, symbol):
	
		self._catchersP[ symbol ].terminate()	
		self._catchersP[ symbol ].kill()
		del self._catchersP[ symbol ]
		print ('BOOK: {} process stopped'.format(symbol))
		if not self._catchersP:
			self._subPrc = False
			self._consumerT.join()
			print('BOOK: consumer thread stopped')
			
	def orderConsumer(self): 
		# catches orders from ordercatcher
	
		while self._subPrc:
			for proc in self._catchersP.values():
				print proc
				
				print "her"
				try:
					line = proc.stdout.readline()
					 print "line:" + line					#	order = json.loads(line[25:]) # shave off twisted timestamp (probably a better way to remove the timestamp...)
					#	print order

				except Exception as e:
					print (e)
		


if __name__ == "__main__":
	book = OrderBook()
	book.startBook("BTC_ETH")
	time.sleep(100)
	book.stopBook("BTC_ETH")
	
