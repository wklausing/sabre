from sabreV5 import Sabre as SabreV5


sabre = SabreV5(verbose=True, abr='throughput', moving_average='ewma', replace='right', abr_osc=False)

sabre.downloadSegment(duration_ms=1000, bandwidth=1000000, latency_ms=30)
sabre.downloadSegment()
sabre.downloadSegment()