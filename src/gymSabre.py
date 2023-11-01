import json
from collections import namedtuple
    
class SabreGym():
    index = -1

    def __init__(self):
        self.network_trace = self.load_json('example/network.json')
        self.NetworkPeriod = namedtuple('NetworkPeriod', 'time bandwidth latency permanent')
        self.network_multiplier = 1
        self.network_trace = [self.NetworkPeriod(time=p['duration_ms'],
                            bandwidth=p['bandwidth_kbps'] *
                            self.network_multiplier,
                            latency=p['latency_ms'],
                            permanent=True)
                for p in self.network_trace]

    def load_json(self, path):
        with open(path) as file:
            obj = json.load(file)
        return obj

    def getNextNetworkCondition(self):
        trace = self.setNetworkCondition(time=1000, bandwidth=100, latency=100)
        return trace
    
    def setNetworkCondition(self, time, bandwidth, latency, permanent=True):
        trace = self.NetworkPeriod(time=time,
                            bandwidth=bandwidth *
                            self.network_multiplier,
                            latency=latency,
                            permanent=permanent)
        return trace

if __name__ == '__main__':
    sabreGym = SabreGym()
    foo = sabreGym.getNextNetworkCondition()
    print(foo)
    foo = sabreGym.getNextNetworkCondition()
    print(foo)




