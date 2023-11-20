from sabreV6 import BolaEnh, Ewma, Util, Replace, SessionInfo
from collections import namedtuple
import math


class NetworkModel:

    DownloadProgress = namedtuple('DownloadProgress',
                                  'index quality '
                                  'size downloaded '
                                  'time time_to_first_bit '
                                  'abandon_to_quality')
    NetworkPeriod = namedtuple('NetworkPeriod', 'time bandwidth latency')

    min_progress_size = 12000
    min_progress_time = 50

    userSetNetworkCondition = False
    userCount = 0


    permanent = True
    network_total_timeTEMP = 0 #self.util.network_total_time
    time_to_nextTEMP = 0 #self.time_to_next
    indexTEMP = 0 # self.index


    def __init__(self, util):
        self.util = util

        self.util.sustainable_quality = None
        self.util.network_total_time = 0
        self.traces = []
        self.index = 0
        self.time_to_next = 0

    def _add_network_condition(self, duration_ms, bandwidth_kbps, latency_ms):
        '''
        Adds a new network condition to self.trace. Will be removed after one use.
        '''
        network_trace = self.NetworkPeriod(time=duration_ms,
                                    bandwidth=bandwidth_kbps *
                                    1,
                                    latency=latency_ms)
        self.traces.append(network_trace)
        self.permanent = True

    def _next_network_period(self):
        '''
        Changes network conditions, according to self.trace
        '''
        self.index += 1
        if self.index >= len(self.traces):
            self.permanent = False
            return
        self.time_to_next = self.traces[self.index].time

        latency_factor = 1 - \
            self.traces[self.index].latency / self.util.manifest.segment_time
        effective_bandwidth = self.traces[self.index].bandwidth * latency_factor

        previous_sustainable_quality = self.util.sustainable_quality
        self.util.sustainable_quality = 0
        for i in range(1, len(self.util.manifest.bitrates)):
            if self.util.manifest.bitrates[i] > effective_bandwidth:
                break
            self.util.sustainable_quality = i
        if (self.util.sustainable_quality != previous_sustainable_quality and
                previous_sustainable_quality != None):
            self.util.advertize_new_network_quality(
                self.util.sustainable_quality, previous_sustainable_quality)
        return True

    def _do_latency_delay(self, delay_units):
        '''
        Return delay time.

        This needs to return false if new network trace is required.
        '''
        total_delay = 0
        while delay_units > 0:
            current_latency = self.traces[self.index].latency
            time = delay_units * current_latency
            if time <= self.time_to_nextTEMP:
                total_delay += time
                self.network_total_timeTEMP += time
                self.time_to_nextTEMP -= time
                delay_units = 0
            else:
                total_delay += self.time_to_nextTEMP
                self.network_total_timeTEMP += self.time_to_nextTEMP
                delay_units -= self.time_to_nextTEMP / current_latency
                self._next_network_period()
                if self.permanent == False: return 0
        return total_delay

    def _do_download(self, size):
        '''
        Return download time
        '''
        total_download_time = 0
        while size > 0:
            current_bandwidth = self.traces[self.index].bandwidth
            if size <= self.time_to_nextTEMP * current_bandwidth:
                time = size / current_bandwidth
                total_download_time += time
                self.network_total_timeTEMP += time
                self.time_to_nextTEMP -= time
                size = 0
            else:
                total_download_time += self.time_to_nextTEMP
                self.network_total_timeTEMP += self.time_to_nextTEMP
                size -= self.time_to_nextTEMP * current_bandwidth
                self._next_network_period()
                if self.permanent == False: break
        return total_download_time
        

    def _do_minimal_latency_delay(self, delay_units, min_time):
        total_delay_units = 0
        total_delay_time = 0
        while delay_units > 0 and min_time > 0:
            current_latency = self.traces[self.index].latency
            time = delay_units * current_latency
            if time <= min_time and time <= self.time_to_nextTEMP:
                units = delay_units
                self.time_to_nextTEMP -= time
                self.network_total_timeTEMP += time
            elif min_time <= self.time_to_nextTEMP:
                # time > 0 implies current_latency > 0
                time = min_time
                units = time / current_latency
                self.time_to_nextTEMP -= time
                self.network_total_timeTEMP += time
            else:
                time = self.time_to_nextTEMP
                units = time / current_latency
                self.network_total_timeTEMP += time
                self._next_network_period()
                if self.permanent == False: break
            total_delay_units += units
            total_delay_time += time
            delay_units -= units
            min_time -= time

        return (total_delay_units, total_delay_time)

    def _do_minimal_download(self, size, min_size, min_time):
        total_size = 0
        total_time = 0
        while size > 0 and (min_size > 0 or min_time > 0):
            current_bandwidth = self.traces[self.index].bandwidth
            if current_bandwidth > 0:
                min_bits = max(min_size, min_time * current_bandwidth)
                bits_to_next = self.time_to_nextTEMP * current_bandwidth
                if size <= min_bits and size <= bits_to_next:
                    bits = size
                    time = bits / current_bandwidth
                    self.time_to_nextTEMP -= time
                    self.network_total_timeTEMP += time
                elif min_bits <= bits_to_next:
                    bits = min_bits
                    time = bits / current_bandwidth
                    # make sure rounding error does not push while loop into endless loop
                    min_size = 0
                    min_time = 0
                    self.time_to_nextTEMP -= time
                    self.network_total_timeTEMP += time
                else:
                    bits = bits_to_next
                    time = self.time_to_nextTEMP
                    self.network_total_timeTEMP += time
                    self._next_network_period()
            else:  # current_bandwidth == 0
                bits = 0
                if min_size > 0 or min_time > self.time_to_next:
                    time = self.time_to_nextTEMP
                    self.util.network_total_time += time
                    self._next_network_period()
                else:
                    time = min_time
                    self.time_to_nextTEMP -= time
                    self.network_total_timeTEMP += time
            
            if self.permanent == False: break

            total_size += bits
            total_time += time
            size -= bits
            min_size -= bits
            min_time -= time
        return (total_size, total_time)

    def delayNet(self, time):
        '''
        I think, it is the delay till the next download. 

        self.util.network_total_time --> network_total_time
        self.time_to_next --> time_to_next
        '''
        while time > self.time_to_nextTEMP:
            time -= self.time_to_nextTEMP
            self.network_total_timeTEMP += self.time_to_nextTEMP
            self._next_network_period()
            if self.permanent == False: break
        self.time_to_nextTEMP -= time
        self.network_total_timeTEMP += time

    def downloadNet(self, size, idx, quality, buffer_level, check_abandon=None):
        '''
        Returns tuple of DownloadProgress.
        '''
        downloadProgress = False # Assuming not enough trace 
        self.network_total_timeTEMP = self.util.network_total_time
        self.time_to_nextTEMP = self.time_to_next

        if size <= 0:# If size is not positive, than return 
            downloadProgress = self.DownloadProgress(index=idx, quality=quality,
                                         size=0, downloaded=0,
                                         time=0, time_to_first_bit=0,
                                         abandon_to_quality=None)
        elif not check_abandon:
            latency = self._do_latency_delay(1)
            time = latency + self._do_download(size)
            downloadProgress = self.DownloadProgress(index=idx, quality=quality,
                                         size=size, downloaded=size,
                                         time=time, time_to_first_bit=latency,
                                         abandon_to_quality=None)
        else:
            total_download_time = 0
            total_download_size = 0
            min_time_to_progress = NetworkModel.min_progress_time
            min_size_to_progress = NetworkModel.min_progress_size

            if NetworkModel.min_progress_size > 0:
                latency = self._do_latency_delay(1)
                if latency == False: return False
                total_download_time += latency
                min_time_to_progress -= total_download_time
                delay_units = 0
            else:
                latency = None
                delay_units = 1

            abandon_quality = None
            while total_download_size < size and abandon_quality == None:

                if delay_units > 0:
                    # NetworkModel.min_progress_size <= 0
                    (units, time) = self._do_minimal_latency_delay(
                        delay_units, min_time_to_progress)
                    total_download_time += time
                    delay_units -= units
                    min_time_to_progress -= time
                    if delay_units <= 0:
                        latency = total_download_time

                if delay_units <= 0:
                    # don't use else to allow fall through
                    (bits, time) = self._do_minimal_download(size - total_download_size,
                                                            min_size_to_progress, min_time_to_progress)
                    total_download_time += time
                    total_download_size += bits
                    # no need to upldate min_[time|size]_to_progress - reset below

                dp = self.DownloadProgress(index=idx, quality=quality,
                                        size=size, downloaded=total_download_size,
                                        time=total_download_time, time_to_first_bit=latency,
                                        abandon_to_quality=None)
                if total_download_size < size:
                    abandon_quality = check_abandon(
                        dp, max(0, buffer_level - total_download_time))
                    if abandon_quality != None:
                        if self.util.verbose:
                            print('%d abandoning %d->%d' %
                                (idx, quality, abandon_quality))
                            print('%d/%d %d(%d)' %
                                (dp.downloaded, dp.size, dp.time, dp.time_to_first_bit))
                    min_time_to_progress = NetworkModel.min_progress_time
                    min_size_to_progress = NetworkModel.min_progress_size

            downloadProgress = self.DownloadProgress(index=idx, quality=quality,
                                        size=size, downloaded=total_download_size,
                                        time=total_download_time, time_to_first_bit=latency,
                                        abandon_to_quality=abandon_quality)
            
        if self.permanent:
            self.util.network_total_time = self.network_total_timeTEMP
            self.time_to_next = self.time_to_nextTEMP
            self.indeFOox = self.indexTEMP

        return downloadProgress


class Sabre():

    abr_list = {}
    abr_list['bolae'] = BolaEnh

    ManifestInfo = namedtuple('ManifestInfo', 'segment_time bitrates utilities segments')
    NetworkPeriod = namedtuple('NetworkPeriod', 'time bandwidth latency permanent')

    average_default = 'ewma'
    average_list = {}
    average_list['ewma'] = Ewma

    util = Util()
    throughput_history = None
    abr = None
    firstSegment = True

    def __init__(
        self,
        abr='bolae',
        abr_basic=False,
        abr_osc=False,
        gamma_p=5,
        half_life=[3, 8],
        max_buffer=25,
        movie='example/movie.json',
        movie_length=None,
        moving_average=average_default,
        network='example/network.json',
        network_multiplier=1,
        no_abandon=False,
        no_insufficient_buffer_rule=False,
        rampup_threshold=None,
        replace='none',
        seek=None,
        verbose=True,
        window_size=[3],
    ):  
        self.no_abandon = no_abandon
        self.seek = seek

        self.util.verbose = verbose

        self.util.buffer_contents = []
        self.util.buffer_fcc = 0
        self.util.pending_quality_up = []

        self.util.rebuffer_event_count = 0
        self.util.rebuffer_time = 0

        self.util.played_utility = 0
        self.util.played_bitrate = 0
        self.util.total_play_time = 0
        self.util.total_bitrate_change = 0
        self.util.total_log_bitrate_change = 0
        self.util.total_reaction_time = 0
        self.util.last_played = None

        self.overestimate_count = 0
        self.overestimate_average = 0
        self.goodestimate_count = 0
        self.goodestimate_average = 0
        self.estimate_average = 0

        self.util.rampup_origin = 0
        self.util.rampup_time = None
        self.util.rampup_threshold = rampup_threshold

        self.util.max_buffer_size = max_buffer * 1000

        self.util.manifest = self.util.load_json(movie)

        bitrates = self.util.manifest['bitrates_kbps']
        utility_offset = 0 - math.log(bitrates[0])  # so utilities[0] = 0
        utilities = [math.log(b) + utility_offset for b in bitrates]
        self.util.manifest = self.ManifestInfo(segment_time=self.util.manifest['segment_duration_ms'],
                                    bitrates=bitrates,
                                    utilities=utilities,
                                    segments=self.util.manifest['segment_sizes_bits'])
        SessionInfo.manifest = self.util.manifest

        self.buffer_size = max_buffer * 1000
        self.gamma_p = gamma_p

        config = {'buffer_size': self.buffer_size,
                'gp': gamma_p,
                'abr_osc': abr_osc,
                'abr_basic': abr_basic,
                'no_ibr': no_insufficient_buffer_rule}
        
        self.abr_list[abr].use_abr_o = abr_osc
        self.abr_list[abr].use_abr_u = not abr_osc
        self.abr = self.abr_list[abr](config, self.util)

        ## Beispiel Trace ##
        trace = self.NetworkPeriod(time=10000000,
                            bandwidth=1000 *
                            1,
                            latency=100,
                            permanent=True)

        self.network = NetworkModel(self.util)

        self.replacer = Replace(1, self.util)

        config = {'window_size': window_size, 'half_life': half_life}
        self.throughput_history = Ewma(config, self.util)

    def downloadSegment(self, segment=None, trace=None):
        ## Bespiel Segment ##
        segment = {'segment_duration_ms': 3000, 
                   'bitrates_kbps': [ 230, 331, 477, 688, 991, 1427, 2056, 2962, 5027, 6000 ], 
                   'segment_sizes_bits': [ 886360, 1180512, 1757888, 2321704, 3515816, 5140704, 7395048, 10097056, 17115584, 20657480 ]}

        if self.firstSegment:
            self.next_segment = 0

        # download first segment
        if self.firstSegment: 
            quality = self.abr.get_first_quality()
            size = segment['segment_sizes_bits'][quality]
            
            self.network._add_network_condition(100,100,100)
            download_metricNew = self.network.downloadNet(size, 0, quality, 0, None)

            while not download_metricNew:
                self.network._add_network_condition(100,100,100)
                download_metricNew = self.network.downloadNet(size, 0, quality, 0, None)

            quit()


            download_time = download_metric.time - download_metric.time_to_first_bit
            self.util.buffer_contents.append(download_metric.quality)
            t = download_metric.size / download_time # t represents throughput per ms
            l = download_metric.time_to_first_bit
            self.throughput_history.push(download_time, t, l)
            self.util.total_play_time += download_metric.time
            self.firstSegment = False
            self.next_segment = 1
        
        


if __name__ == '__main__':
    sabre = Sabre()
    sabre.downloadSegment()