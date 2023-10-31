import json
from collections import namedtuple



firstRequest = True
index = 0
    
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
        self.index += 1
        if self.index >= len(self.network_trace):
            self.index = 0

        trace = self.network_trace[self.index]
        
        return trace

if __name__ == '__main__':
    sabreGym = SabreGym()
    foo = sabreGym.getNextNetworkCondition()
    print(foo)
    foo = sabreGym.getNextNetworkCondition()
    print(foo)


