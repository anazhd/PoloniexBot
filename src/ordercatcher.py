
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
from twisted.internet.defer import inlineCallbacks

class OrderCatcher(ApplicationSession):
        def __init__(self, symbol, config=None):
                pass
        @inlineCallbacks
        def onJoin(self, details):

                def onOrder(*args, **kw):
                        print json.dumps(args)
                yield self.subscribe(onOrder, symbol)
