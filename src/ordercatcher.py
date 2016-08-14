
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
from twisted.internet.defer import inlineCallbacks
import json
import sys

global symbol

symbol = sys.argv[1]
print symbol

def onOrder(*args, **kw):
	print(json.dumps(args))

class OrderCatcher(ApplicationSession):

	def __init__(self,  config=None):
        	ApplicationSession.__init__(self, config)
              

        @inlineCallbacks
        def onJoin(self, details):
		yield self.subscribe(onOrder, symbol)
	 

if __name__ == "__main__":
	subscriber = ApplicationRunner(u"wss://api.poloniex.com:443", u"realm1")
        subscriber.run( OrderCatcher )
