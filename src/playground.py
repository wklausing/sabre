from sabreV4 import Sabre as SabreV4


sabre = SabreV4(verbose=True, abr='throughput', moving_average='ewma', replace='right', abr_osc=False)

sabre.downloadSegment()
sabre.downloadSegment()
sabre.downloadSegment()