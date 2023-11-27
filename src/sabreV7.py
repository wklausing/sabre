from sabreV6 import BolaEnh, Ewma, Util, Replace, SessionInfo, SlidingWindow, Bola, ThroughputRule, Dynamic, DynamicDash, Bba
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

    permanent = False # If false does also mean that the network trace is empty
    network_total_timeTEMP = 0 #self.util.network_total_time

    def __init__(self, util, duration_ms=1000, bandwidth_kbps=100, latency_ms=100):
        self.util = util

        self.util.sustainable_quality = None
        self.util.network_total_time = 0
        self.traces = []
        self.index = -1
        self.indexTEMP = -1
        self.time_to_next = 0
        self.time_to_nextTEMP = 0
        # self._add_network_condition(duration_ms, bandwidth_kbps, latency_ms)
        # if self._next_network_period() == False:
        #     print('Missing initial network condition')
        #     return False

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
        self.indexTEMP += 1
        if self.indexTEMP >= len(self.traces):
            self.permanent = False
            self.indexTEMP -= 1
            return False
        self.time_to_nextTEMP = self.traces[self.indexTEMP].time

        latency_factor = 1 - \
            self.traces[self.indexTEMP].latency / self.util.manifest.segment_time
        effective_bandwidth = self.traces[self.indexTEMP].bandwidth * latency_factor

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
            current_latency = self.traces[self.indexTEMP].latency # 200
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
        while size >= 0:
            current_bandwidth = self.traces[self.indexTEMP].bandwidth
            if size <= self.time_to_nextTEMP * current_bandwidth:
                time = size / current_bandwidth #1481.8 = 296360 / 200 | 1531.8 = 306360.0 / 200
                total_download_time += time
                self.network_total_timeTEMP += time
                self.time_to_nextTEMP -= time
                break
            else:
                total_download_time += self.time_to_nextTEMP
                self.network_total_timeTEMP += self.time_to_nextTEMP
                size -= self.time_to_nextTEMP * current_bandwidth
                #print('size', size, 'time_to_nextTEMP', self.time_to_nextTEMP, 'current_bandwidth', current_bandwidth)
                self._next_network_period()
                if self.permanent == False: break
        return total_download_time
        

    def _do_minimal_latency_delay(self, delay_units, min_time):
        total_delay_units = 0
        total_delay_time = 0
        while delay_units > 0 and min_time > 0:
            current_latency = self.traces[self.indexTEMP].latency
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
            current_bandwidth = self.traces[self.indexTEMP].bandwidth
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

        if self.permanent == False: return False

        while time > self.time_to_nextTEMP:
            time -= self.time_to_nextTEMP
            self.network_total_timeTEMP += self.time_to_nextTEMP
            self._next_network_period()
            if self.permanent == False: break
        self.time_to_nextTEMP -= time
        self.network_total_timeTEMP += time

        self.util.network_total_time = self.network_total_timeTEMP
        self.time_to_next = self.time_to_nextTEMP

    def downloadNet(self, size, idx, quality, buffer_level, check_abandon=None):
        '''
        Returns tuple of DownloadProgress.
        '''
        if self.permanent == False: return False

        downloadProgress = False # Assuming not enough trace 
        self.network_total_timeTEMP = self.util.network_total_time
        self.time_to_nextTEMP = self.time_to_next
        self.indexTEMP = self.index

        print('self.util.network_total_time', self.util.network_total_time, 'self.time_to_next', self.time_to_next, 'self.index', self.index)
        #Ist: self.util.network_total_time 0 self.time_to_next 0 self.index -1

        if size <= 0:# If size is not positive, than return 
            downloadProgress = self.DownloadProgress(index=idx, quality=quality,
                                         size=0, downloaded=0,
                                         time=0, time_to_first_bit=0,
                                         abandon_to_quality=None)
        elif not check_abandon or (NetworkModel.min_progress_time <= 0 and
                                 NetworkModel.min_progress_size <= 0):
            
            latency = self._do_latency_delay(1) # 100
            if self.permanent == False: return False

            time = latency + self._do_download(size)# 5481.8
            if self.permanent == False: return False
            
            return self.DownloadProgress(index=idx, quality=quality,
                                         size=size, downloaded=size,
                                         time=time, time_to_first_bit=latency,
                                         abandon_to_quality=None)
        else:
            total_download_time = 0
            total_download_size = 0
            min_time_to_progress = NetworkModel.min_progress_time
            min_size_to_progress = NetworkModel.min_progress_size

            if NetworkModel.min_progress_size > 0:
                latency = self._do_latency_delay(1) # Sollte 200 sein, ist aber 100
                total_download_time += latency # 200
                min_time_to_progress -= total_download_time # -150
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
            self.index = self.indexTEMP

        return downloadProgress


class Sabre():

    average_default = 'ewma'
    average_list = {}
    average_list['ewma'] = Ewma
    average_list['sliding'] = SlidingWindow

    abr_default = 'bolae'
    abr_list = {}
    abr_list['bola'] = Bola
    abr_list['bolae'] = BolaEnh
    abr_list['throughput'] = ThroughputRule
    abr_list['dynamic'] = Dynamic
    abr_list['dynamicdash'] = DynamicDash
    abr_list['bba'] = Bba

    ManifestInfo = namedtuple('ManifestInfo', 'segment_time bitrates utilities segments')
    NetworkPeriod = namedtuple('NetworkPeriod', 'time bandwidth latency permanent')

    util = Util()
    throughput_history = None
    abr = None
    firstSegment = True
    next_segment = 0

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
        window_size=[3]
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

        self.network = NetworkModel(self.util)

        self.replacer = Replace(1, self.util)

        config = {'window_size': window_size, 'half_life': half_life}
        self.throughput_history = Ewma(config, self.util)

    def downloadSegment(self, trace=None, lastSegment=False):

        # Final playout of buffer at the end.
        if self.next_segment == len(self.util.manifest.segments):
            self.util.playout_buffer()
            print('DONE')
            return

        # Download first segment
        if self.firstSegment:
            quality = self.abr.get_first_quality()
            size = self.util.manifest.segments[0][quality]

            download_metric = self.network.downloadNet(size, 0, quality, 0, None)
            if download_metric == False: return False

            download_time = download_metric.time - download_metric.time_to_first_bit
            self.util.buffer_contents.append(download_metric.quality)
            t = download_metric.size / download_time # t represents throughput per ms
            l = download_metric.time_to_first_bit
            self.throughput_history.push(download_time, t, l)
            self.util.total_play_time += download_metric.time # 5481.8
            self.firstSegment = False
            self.next_segment = 1
            self.abandoned_to_quality = None
        else:
            # Download rest of segments

            # do we have space for a new segment on the buffer?
            full_delay = self.util.get_buffer_level() + self.util.manifest.segment_time - self.buffer_size

            if full_delay > 0:
                self.network.delayNet(full_delay)
                self.util.deplete_buffer(full_delay)
                self.abr.report_delay(full_delay)
                if self.util.verbose:
                    print('full buffer delay %d bl=%d' %
                        (full_delay, self.util.get_buffer_level()))

            if self.abandoned_to_quality == None:
                (quality, delay) = self.abr.get_quality_delay(self.next_segment)
                replace = self.replacer.check_replace(quality)
            else:
                (quality, delay) = (self.abandoned_to_quality, 0)
                replace = None
                self.abandon_to_quality = None

            if replace != None:
                delay = 0
                current_segment = self.next_segment + replace
                check_abandon = self.replacer.check_abandon
            else:
                current_segment = self.next_segment
                check_abandon = self.abr.check_abandon
            if self.no_abandon:
                check_abandon = None

            size = self.util.manifest.segments[current_segment][quality]

            if delay > 0:
                self.network.delayNet(delay)
                self.util.deplete_buffer(delay)
                if self.util.verbose:
                    print('abr delay %d bl=%d' % (delay, self.util.get_buffer_level()))
            
            print('size', size, 'current_segment', current_segment, 'quality', quality, 'buffer_level', self.util.get_buffer_level())
            # Here ist alles richtig
            download_metric = self.network.downloadNet(size, current_segment, quality,
                                            self.util.get_buffer_level(), check_abandon)

            self.util.deplete_buffer(download_metric.time) #Is 5481.8, should be 5631.8

            # Update buffer with new download
            if replace == None:
                if download_metric.abandon_to_quality == None:
                    self.util.buffer_contents += [quality]
                    self.next_segment += 1
                else:
                    self.abandon_to_quality = download_metric.abandon_to_quality
            else:
                if download_metric.abandon_to_quality == None:
                    if self.util.get_buffer_level() + self.util.manifest.segment_time * replace >= 0:
                        self.util.buffer_contents[replace] = quality
                    else:
                        print('WARNING: too late to replace')
                        pass
                else:
                    pass
                # else: do nothing because segment abandonment does not suggest new download

            if self.util.verbose:
                print('->%d' % self.util.get_buffer_level())

            self.abr.report_download(download_metric, replace != None)

            # calculate throughput and latency
            download_time = download_metric.time - download_metric.time_to_first_bit
            t = download_metric.downloaded / download_time
            l = download_metric.time_to_first_bit

            # check accuracy of throughput estimate
            if self.util.throughput > t:
                self.overestimate_count += 1
                self.overestimate_average += (self.util.throughput - t -
                                        self.overestimate_average) / self.overestimate_count
            else:
                self.goodestimate_count += 1
                self.goodestimate_average += (t - self.util.throughput -
                                        self.goodestimate_average) / self.goodestimate_count
            self.estimate_average += ((self.util.throughput - t - self.estimate_average) /
                                (self.overestimate_count + self.goodestimate_count))

            # update throughput estimate
            if download_metric.abandon_to_quality == None:
                self.throughput_history.push(download_time, t, l)

            # loop while next_segment < len(manifest.segments)
        
        # Is 10963.599999999999 but should be 11113.599999999999. Missing 150
        to_time_average = 1 / (self.util.total_play_time / self.util.manifest.segment_time)

        result = {}
        result['buffer_size'] = self.buffer_size
        result['time_average_played_bitrate'] = 1 / (self.util.total_play_time / self.util.manifest.segment_time)#10963.599999999999 / 3000
        result['time_average_bitrate_change'] = self.util.total_bitrate_change * to_time_average
        result['time_average_rebuffer_events'] = self.util.rebuffer_event_count * to_time_average
        return result


if __name__ == '__main__':
    sabre = Sabre(verbose=False, abr='throughput', moving_average='ewma', replace='right', abr_osc=False)

    foo = False
    i = 0
    while foo == False:
        foo = sabre.downloadSegment()
        if i % 2 == 0 and not foo:
            sabre.network._add_network_condition(1000,100,100)
        elif not foo:
            sabre.network._add_network_condition(2000,200,200)
        i += 1
    seg = {'buffer_size': 25000, 'time_average_played_bitrate': 0.5472654967346492, 'time_average_bitrate_change': 0.0, 'time_average_rebuffer_events': 0.0}
    if foo == seg:
        print('Seg1 is correct')
    else:
        print('Seg1 wrong')
        quit()

    foo = False
    i = 0
    while foo == False:
        foo = sabre.downloadSegment()
        if i % 2 == 0 and not foo:
            sabre.network._add_network_condition(1000,100,100)
        elif not foo:
            sabre.network._add_network_condition(2000,200,200)
        i += 1
    print(foo)
    seg2 = {'buffer_size': 25000, 'time_average_played_bitrate': 0.2699395335444861, 'time_average_bitrate_change': 0.0, 'time_average_rebuffer_events': 0.2699395335444861}
    if foo == seg2:
        print('Seg2 is correct')
    else:
        print('Seg2 wrong')
        quit()


        # {'buffer_size': 25000, 'time_average_played_bitrate': 0.5472654967346492, 'time_average_bitrate_change': 0.0, 'time_average_rebuffer_events': 0.0}
        # {'buffer_size': 25000, 'time_average_played_bitrate': 0.2699395335444861, 'time_average_bitrate_change': 0.0, 'time_average_rebuffer_events': 0.2699395335444861}