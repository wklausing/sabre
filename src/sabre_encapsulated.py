import json
import math
import os
from importlib.machinery import SourceFileLoader
from collections import namedtuple
from enum import Enum

class Util:

    def load_json(self, path):
        with open(path) as file:
            obj = json.load(file)
        return obj

    def get_buffer_level(self):
        global manifest
        global buffer_contents
        global buffer_fcc

        return manifest.segment_time * len(buffer_contents) - buffer_fcc

    def deplete_buffer(self, time):
        global manifest
        global buffer_contents
        global buffer_fcc
        global rebuffer_event_count
        global rebuffer_time
        global played_utility
        global played_bitrate
        global total_play_time
        global total_bitrate_change
        global total_log_bitrate_change
        global last_played

        global rampup_origin
        global rampup_time
        global rampup_threshold
        global sustainable_quality

        if len(buffer_contents) == 0:
            rebuffer_time += time
            total_play_time += time
            return

        if buffer_fcc > 0:
            # first play any partial chunk left

            if time + buffer_fcc < manifest.segment_time:
                buffer_fcc += time
                total_play_time += time
                return

            time -= manifest.segment_time - buffer_fcc
            total_play_time += manifest.segment_time - buffer_fcc
            buffer_contents.pop(0)
            buffer_fcc = 0

        # buffer_fcc == 0 if we're here

        while time > 0 and len(buffer_contents) > 0:

            quality = buffer_contents[0]
            played_utility += manifest.utilities[quality]
            played_bitrate += manifest.bitrates[quality]
            if quality != last_played and last_played != None:
                total_bitrate_change += abs(manifest.bitrates[quality] -
                                            manifest.bitrates[last_played])
                total_log_bitrate_change += abs(math.log(manifest.bitrates[quality] /
                                                        manifest.bitrates[last_played]))
            last_played = quality

            if rampup_time == None:
                rt = sustainable_quality if rampup_threshold == None else rampup_threshold
                if quality >= rt:
                    rampup_time = total_play_time - rampup_origin

            # bookkeeping to track reaction time to increased bandwidth
            for p in pending_quality_up:
                if len(p) == 2 and quality >= p[1]:
                    p.append(total_play_time)

            if time >= manifest.segment_time:
                buffer_contents.pop(0)
                total_play_time += manifest.segment_time
                time -= manifest.segment_time
            else:
                buffer_fcc = time
                total_play_time += time
                time = 0

        if time > 0:
            rebuffer_time += time
            total_play_time += time
            rebuffer_event_count += 1

        self.process_quality_up(total_play_time)

    def playout_buffer(self):
        global buffer_contents
        global buffer_fcc

        self.deplete_buffer(self.get_buffer_level())

        # make sure no rounding error
        del buffer_contents[:]
        buffer_fcc = 0

    def process_quality_up(self, now):
        global max_buffer_size
        global pending_quality_up
        global total_reaction_time

        # check which switches can be processed

        cutoff = now - max_buffer_size
        while len(pending_quality_up) > 0 and pending_quality_up[0][0] < cutoff:
            p = pending_quality_up.pop(0)
            if len(p) == 2:
                reaction = max_buffer_size
            else:
                reaction = min(max_buffer_size, p[2] - p[0])
            print('\n[%d] reaction time: %d' % (now, reaction))
            total_reaction_time += reaction

    def advertize_new_network_quality(self, quality, previous_quality):
        global max_buffer_size
        global network_total_time
        global pending_quality_up
        global buffer_contents

        # bookkeeping to track reaction time to increased bandwidth

        # process any previous quality up switches that have "matured"
        self.process_quality_up(network_total_time)

        # mark any pending switch up done if new quality switches back below its quality
        for p in pending_quality_up:
            if len(p) == 2 and p[1] > quality:
                p.append(network_total_time)
        #pending_quality_up = [p for p in pending_quality_up if p[1] >= quality]

        # filter out switches which are not upwards (three separate checks)
        if quality <= previous_quality:
            return
        for q in buffer_contents:
            if quality <= q:
                return
        for p in pending_quality_up:
            if quality <= p[1]:
                return

        # valid quality up switch
        pending_quality_up.append([network_total_time, quality])

class NetworkModel:

    min_progress_size = 12000
    min_progress_time = 50

    def __init__(self, network_trace, util):
        global sustainable_quality
        global network_total_time

        self.util = util

        sustainable_quality = None
        network_total_time = 0
        self.trace = network_trace
        self.index = -1
        self.time_to_next = 0
        self.next_network_period()

    def next_network_period(self):
        global manifest
        global sustainable_quality
        global network_total_time

        self.index += 1
        if self.index == len(self.trace):
            self.index = 0
        self.time_to_next = self.trace[self.index].time

        latency_factor = 1 - self.trace[self.index].latency / manifest.segment_time
        effective_bandwidth = self.trace[self.index].bandwidth * latency_factor

        previous_sustainable_quality = sustainable_quality
        sustainable_quality = 0
        for i in range(1, len(manifest.bitrates)):
            if manifest.bitrates[i] > effective_bandwidth:
                break
            sustainable_quality = i
        if (sustainable_quality != previous_sustainable_quality and
            previous_sustainable_quality != None):
            self.util.advertize_new_network_quality(sustainable_quality, previous_sustainable_quality)

        if verbose:
            print('[%d] Network: %d,%d  (q=%d: bitrate=%d)' %
                  (round(network_total_time),
                   self.trace[self.index].bandwidth, self.trace[self.index].latency,
                   sustainable_quality, manifest.bitrates[sustainable_quality]))

    # return delay time
    def do_latency_delay(self, delay_units):
        global network_total_time

        total_delay = 0
        while delay_units > 0:
            current_latency = self.trace[self.index].latency
            time = delay_units * current_latency
            if time <= self.time_to_next:
                total_delay += time
                network_total_time += time
                self.time_to_next -= time
                delay_units = 0
            else:
                # time > self.time_to_next implies current_latency > 0
                total_delay += self.time_to_next
                network_total_time += self.time_to_next
                delay_units -= self.time_to_next / current_latency
                self.next_network_period()
        return total_delay

    # return download time
    def do_download(self, size):
        global network_total_time
        total_download_time = 0
        while size > 0:
            current_bandwidth = self.trace[self.index].bandwidth
            if size <= self.time_to_next * current_bandwidth:
                # current_bandwidth > 0
                time = size / current_bandwidth
                total_download_time += time
                network_total_time += time
                self.time_to_next -= time
                size = 0
            else:
                total_download_time += self.time_to_next
                network_total_time += self.time_to_next
                size -= self.time_to_next * current_bandwidth
                self.next_network_period()
        return total_download_time

    def do_minimal_latency_delay(self, delay_units, min_time):
        global network_total_time
        total_delay_units = 0
        total_delay_time = 0
        while delay_units > 0 and min_time > 0:
            current_latency = self.trace[self.index].latency
            time = delay_units * current_latency
            if time <= min_time and time <= self.time_to_next:
                units = delay_units
                self.time_to_next -= time
                network_total_time += time
            elif min_time <= self.time_to_next:
                # time > 0 implies current_latency > 0
                time = min_time
                units = time / current_latency
                self.time_to_next -= time
                network_total_time += time
            else:
                time = self.time_to_next
                units = time / current_latency
                network_total_time += time
                self.next_network_period()
            total_delay_units += units
            total_delay_time += time
            delay_units -= units
            min_time -= time
        return (total_delay_units, total_delay_time)

    def do_minimal_download(self, size, min_size, min_time):
        global network_total_time
        total_size = 0
        total_time = 0
        while size > 0 and (min_size > 0 or min_time > 0):
            current_bandwidth = self.trace[self.index].bandwidth
            if current_bandwidth > 0:
                min_bits = max(min_size, min_time * current_bandwidth)
                bits_to_next = self.time_to_next * current_bandwidth
                if size <= min_bits and size <= bits_to_next:
                    bits = size
                    time = bits / current_bandwidth
                    self.time_to_next -= time
                    network_total_time += time
                elif min_bits <= bits_to_next:
                    bits = min_bits
                    time = bits / current_bandwidth
                    # make sure rounding error does not push while loop into endless loop
                    min_size = 0
                    min_time = 0
                    self.time_to_next -= time
                    network_total_time += time
                else:
                    bits = bits_to_next
                    time = self.time_to_next
                    network_total_time += time
                    self.next_network_period()
            else: # current_bandwidth == 0
                bits = 0
                if min_size > 0 or min_time > self.time_to_next:
                    time = self.time_to_next
                    network_total_time += time
                    self.next_network_period()
                else:
                    time = min_time
                    self.time_to_next -= time
                    network_total_time += time
            total_size += bits
            total_time += time
            size -= bits
            min_size -= bits
            min_time -= time
        return (total_size, total_time)

    def delay(self, time):
        global network_total_time
        while time > self.time_to_next:
            time -= self.time_to_next
            network_total_time += self.time_to_next
            self.next_network_period()
        self.time_to_next -= time
        network_total_time += time

    def download(self, size, idx, quality, buffer_level, check_abandon = None):

        DownloadProgress = namedtuple('DownloadProgress',
                              'index quality '
                              'size downloaded '
                              'time time_to_first_bit '
                              'abandon_to_quality')

        if size <= 0:
            return DownloadProgress(index = idx, quality = quality,
                                    size = 0, downloaded = 0,
                                    time = 0, time_to_first_bit = 0,
                                    abandon_to_quality = None)

        if not check_abandon or (NetworkModel.min_progress_time <= 0 and
                                 NetworkModel.min_progress_size <= 0):
            latency = self.do_latency_delay(1)
            time = latency + self.do_download(size)
            return DownloadProgress(index = idx, quality = quality,
                                    size = size, downloaded = size,
                                    time = time, time_to_first_bit = latency,
                                    abandon_to_quality = None)

        total_download_time = 0
        total_download_size = 0
        min_time_to_progress = NetworkModel.min_progress_time
        min_size_to_progress = NetworkModel.min_progress_size

        if NetworkModel.min_progress_size > 0:
            latency = self.do_latency_delay(1)
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
                (units, time) = self.do_minimal_latency_delay(delay_units, min_time_to_progress)
                total_download_time += time
                delay_units -= units
                min_time_to_progress -= time
                if delay_units <= 0:
                    latency = total_download_time

            if delay_units <= 0:
                # don't use else to allow fall through
                (bits, time) = self.do_minimal_download(size - total_download_size,
                                                        min_size_to_progress, min_time_to_progress)
                total_download_time += time
                total_download_size += bits
                # no need to upldate min_[time|size]_to_progress - reset below

            dp = DownloadProgress(index = idx, quality = quality,
                                  size = size, downloaded = total_download_size,
                                  time = total_download_time, time_to_first_bit = latency,
                                  abandon_to_quality = None)
            if total_download_size < size:
                abandon_quality = check_abandon(dp, max(0, buffer_level - total_download_time))
                if abandon_quality != None:
                    if verbose:
                        print('%d abandoning %d->%d' % (idx, quality, abandon_quality))
                        print('%d/%d %d(%d)' %
                              (dp.downloaded, dp.size, dp.time, dp.time_to_first_bit))
                min_time_to_progress = NetworkModel.min_progress_time
                min_size_to_progress = NetworkModel.min_progress_size

        return DownloadProgress(index = idx, quality = quality,
                                size = size, downloaded = total_download_size,
                                time = total_download_time, time_to_first_bit = latency,
                                abandon_to_quality = abandon_quality)

class ThroughputHistory:
    def __init__(self, config):
        pass
    def push(self, time, tput, lat):
        raise NotImplementedError

class SessionInfo:

    def __init__(self):
        pass

    def get_throughput(self):
        global throughput
        return throughput

    def get_buffer_contents(self):
        global buffer_contents
        return buffer_contents[:]

session_info = SessionInfo()

class SlidingWindow(ThroughputHistory):

    default_window_size = [3]
    max_store = 20

    def __init__(self, config):
        global throughput
        global latency

        if 'window_size' in config and config['window_size'] != None:
            self.window_size = config['window_size']
        else:
            self.window_size = SlidingWindow.default_window_size

        throughput = None
        latency = None

        self.last_throughputs = []
        self.last_latencies = []

    def push(self, time, tput, lat):
        global throughput
        global latency

        self.last_throughputs += [tput]
        self.last_throughputs = self.last_throughputs[-SlidingWindow.max_store:]

        self.last_latencies += [lat]
        self.last_latencies = self.last_latencies[-SlidingWindow.max_store:]

        tput = None
        lat = None
        for ws in self.window_size:
            sample = self.last_throughputs[-ws:]
            t = sum(sample) / len(sample)
            tput = t if tput == None else min(tput, t) # conservative min
            sample = self.last_latencies[-ws:]
            l = sum(sample) / len(sample)
            lat = l if lat == None else max(lat, l) # conservative max
        throughput = tput
        latency = lat

class Ewma(ThroughputHistory):

    # for throughput:
    default_half_life = [8000, 3000]

    def __init__(self, config):
        global throughput
        global latency

        # TODO: init somewhere else?
        throughput = None
        latency = None

        if 'half_life' in config and config['half_life'] != None:
            self.half_life = [h * 1000 for h in config['half_life']]
        else:
            self.half_life = Ewma.default_half_life

        self.latency_half_life = [h / manifest.segment_time for h in self.half_life]

        self.throughput = [0] * len(self.half_life)
        self.weight_throughput = 0
        self.latency = [0] * len(self.half_life)
        self.weight_latency = 0

    def push(self, time, tput, lat):
        global throughput
        global latency

        for i in range(len(self.half_life)):
            alpha = math.pow(0.5, time / self.half_life[i])
            self.throughput[i] = alpha * self.throughput[i] + (1 - alpha) * tput
            alpha = math.pow(0.5, 1 / self.latency_half_life[i])
            self.latency[i] = alpha * self.latency[i] + (1 - alpha) * lat

        self.weight_throughput += time
        self.weight_latency += 1

        tput = None
        lat = None
        for i in range(len(self.half_life)):
            zero_factor = 1 - math.pow(0.5, self.weight_throughput / self.half_life[i])
            t = self.throughput[i] / zero_factor
            tput = t if tput == None else min(tput, t)  # conservative case is min
            zero_factor = 1 - math.pow(0.5, self.weight_latency / self.latency_half_life[i])
            l = self.latency[i] / zero_factor
            lat = l if lat == None else max(lat, l) # conservative case is max
        throughput = tput
        latency = lat

average_list = {}
average_list['sliding'] = SlidingWindow
average_list['ewma'] = Ewma
average_default = 'ewma'

class Abr:

    session = session_info

    def __init__(self, config, util):
        pass
    def get_quality_delay(self, segment_index):
        raise NotImplementedError
    def get_first_quality(self):
        return 0
    def report_delay(self, delay):
        pass
    def report_download(self, metrics, is_replacment):
        pass
    def report_seek(self, where):
        pass
    def check_abandon(self, progress, buffer_level):
        return None

    def quality_from_throughput(self, tput):
        global manifest
        global throughput
        global latency

        p = manifest.segment_time

        quality = 0
        while (quality + 1 < len(manifest.bitrates) and
               latency + p * manifest.bitrates[quality + 1] / tput <= p):
            quality += 1
        return quality

class Bola(Abr):

    def __init__(self, config, util):
        global verbose
        global manifest

        self.util = util

        utility_offset = -math.log(manifest.bitrates[0]) # so utilities[0] = 0
        self.utilities = [math.log(b) + utility_offset for b in manifest.bitrates]

        self.gp = config['gp']
        self.buffer_size = config['buffer_size']
        self.abr_osc = config['abr_osc']
        self.abr_basic = config['abr_basic']
        self.Vp = (self.buffer_size - manifest.segment_time) / (self.utilities[-1] + self.gp)

        self.last_seek_index = 0 # TODO
        self.last_quality = 0

        if verbose:
            for q in range(len(manifest.bitrates)):
                b = manifest.bitrates[q]
                u = self.utilities[q]
                l = self.Vp * (self.gp + u)
                if q == 0:
                    print('%d %d' % (q, l))
                else:
                    qq = q - 1
                    bb = manifest.bitrates[qq]
                    uu = self.utilities[qq]
                    ll = self.Vp * (self.gp + (b * uu - bb * u) / (b - bb))
                    print('%d %d    <- %d %d' % (q, l, qq, ll))

    def quality_from_buffer(self):
        level = self.util.get_buffer_level()
        quality = 0
        score = None
        for q in range(len(manifest.bitrates)):
            s = ((self.Vp * (self.utilities[q] + self.gp) - level) / manifest.bitrates[q])
            if score == None or s > score:
                quality = q
                score = s
        return quality

    def get_quality_delay(self, segment_index):
        global manifest
        global throughput

        if not self.abr_basic:
            t = min(segment_index - self.last_seek_index, len(manifest.segments) - segment_index)
            t = max(t / 2, 3)
            t = t * manifest.segment_time
            buffer_size = min(self.buffer_size, t)
            self.Vp = (buffer_size - manifest.segment_time) / (self.utilities[-1] + self.gp)

        quality = self.quality_from_buffer()
        delay = 0

        if quality > self.last_quality:
            quality_t = self.quality_from_throughput(throughput)
            if quality <= quality_t:
                delay = 0
            elif self.last_quality > quality_t:
                quality = self.last_quality
                delay = 0
            else:
                if not self.abr_osc:
                    quality = quality_t + 1
                    delay = 0
                else:
                    quality = quality_t
                    # now need to calculate delay
                    b = manifest.bitrates[quality]
                    u = self.utilities[quality]
                    #bb = manifest.bitrates[quality + 1]
                    #uu = self.utilities[quality + 1]
                    #l = self.Vp * (self.gp + (bb * u - b * uu) / (bb - b))
                    l = self.Vp * (self.gp + u) ##########
                    delay = max(0, self.util.get_buffer_level() - l)
                    if quality == len(manifest.bitrates) - 1:
                        delay = 0
                    # delay = 0 ###########

        self.last_quality = quality
        return (quality, delay)

    def report_seek(self, where):
        # TODO: seek properly
        global manifest
        self.last_seek_index = math.floor(where / manifest.segment_time)

    def check_abandon(self, progress, buffer_level):
        global manifest

        if self.abr_basic:
            return None

        remain = progress.size - progress.downloaded
        if progress.downloaded <= 0 or remain <= 0:
            return None

        abandon_to = None
        score = (self.Vp * (self.gp + self.utilities[progress.quality]) - buffer_level) / remain
        if score < 0:
            return # TODO: check

        for q in range(progress.quality):
            other_size = progress.size * manifest.bitrates[q] / manifest.bitrates[progress.quality]
            other_score = (self.Vp * (self.gp + self.utilities[q]) - buffer_level) / other_size
            if other_size < remain and other_score > score:
                # check size: see comment in BolaEnh.check_abandon()
                score = other_score
                abandon_to = q

        if abandon_to != None:
            self.last_quality = abandon_to

        return abandon_to

class BolaEnh(Abr):

    minimum_buffer = 10000
    minimum_buffer_per_level = 2000
    low_buffer_safety_factor = 0.5
    low_buffer_safety_factor_init = 0.9

    class State(Enum):
        STARTUP = 1
        STEADY = 2

    def __init__(self, config, util):
        global verbose
        global manifest

        self.util = util

        config_buffer_size = config['buffer_size']
        self.abr_osc = config['abr_osc']
        self.no_ibr = config['no_ibr']

        utility_offset = 1 - math.log(manifest.bitrates[0]) # so utilities[0] = 1
        self.utilities = [math.log(b) + utility_offset for b in manifest.bitrates]

        if self.no_ibr:
            self.gp = config['gp'] - 1 # to match BOLA Basic
            buffer = config['buffer_size']
            self.Vp = (buffer - manifest.segment_time) / (self.utilities[-1] + self.gp)
        else:
            buffer = BolaEnh.minimum_buffer
            buffer += BolaEnh.minimum_buffer_per_level * len(manifest.bitrates)
            buffer = max(buffer, config_buffer_size)
            print(buffer)
            self.gp = (self.utilities[-1] - 1) / (buffer / BolaEnh.minimum_buffer - 1)
            self.Vp = BolaEnh.minimum_buffer / self.gp
            #equivalently:
            #self.Vp = (buffer - BolaEnh.minimum_buffer) / (math.log(manifest.bitrates[-1] / manifest.bitrates[0]))
            #self.gp = BolaEnh.minimum_buffer / self.Vp

        self.state = BolaEnh.State.STARTUP
        self.placeholder = 0
        self.last_quality = 0

        if verbose:
            for q in range(len(manifest.bitrates)):
                b = manifest.bitrates[q]
                u = self.utilities[q]
                l = self.Vp * (self.gp + u)
                if q == 0:
                    print('%d %d' % (q, l))
                else:
                    qq = q - 1
                    bb = manifest.bitrates[qq]
                    uu = self.utilities[qq]
                    ll = self.Vp * (self.gp + (b * uu - bb * u) / (b - bb))
                    print('%d %d    <- %d %d' % (q, l, qq, ll))

    def quality_from_buffer(self, level):
        if level == None:
            level = self.util.get_buffer_level()
        quality = 0
        score = None
        for q in range(len(manifest.bitrates)):
            s = ((self.Vp * (self.utilities[q] + self.gp) - level) / manifest.bitrates[q])
            if score == None or s > score:
                quality = q
                score = s
        return quality

    def quality_from_buffer_placeholder(self):
        return self.quality_from_buffer(self.util.get_buffer_level() + self.placeholder)

    def min_buffer_for_quality(self, quality):
        global manifest

        bitrate = manifest.bitrates[quality]
        utility = self.utilities[quality]

        level = 0
        for q in range(quality):
            # for each bitrates[q] less than bitrates[quality],
            # BOLA should prefer bitrates[quality]
            # (unless bitrates[q] has higher utility)
            if self.utilities[q] < self.utilities[quality]:
                b = manifest.bitrates[q]
                u = self.utilities[q]
                l = self.Vp * (self.gp + (bitrate * u - b * utility) / (bitrate - b))
                level = max(level, l)
        return level

    def max_buffer_for_quality(self, quality):
        return self.Vp * (self.utilities[quality] + self.gp)

    def get_quality_delay(self, segment_index):
        global buffer_contents
        global buffer_fcc
        global throughput

        buffer_level = self.util.get_buffer_level()

        if self.state == BolaEnh.State.STARTUP:
            if throughput == None:
                return (self.last_quality, 0)
            self.state = BolaEnh.State.STEADY
            self.ibr_safety = BolaEnh.low_buffer_safety_factor_init
            quality = self.quality_from_throughput(throughput)
            self.placeholder = self.min_buffer_for_quality(quality) - buffer_level
            self.placeholder = max(0, self.placeholder)
            return (quality, 0)

        quality = self.quality_from_buffer_placeholder()
        quality_t = self.quality_from_throughput(throughput)
        if quality > self.last_quality and quality > quality_t:
            quality = max(self.last_quality, quality_t)
            if not self.abr_osc:
                quality += 1

        max_level = self.max_buffer_for_quality(quality)

        ################
        if quality > 0:
            q = quality
            b = manifest.bitrates[q]
            u = self.utilities[q]
            qq = q - 1
            bb = manifest.bitrates[qq]
            uu = self.utilities[qq]
            #max_level = self.Vp * (self.gp + (b * uu - bb * u) / (b - bb))
        ################

        delay = buffer_level + self.placeholder - max_level
        if delay > 0:
            if delay <= self.placeholder:
                self.placeholder -= delay
                delay = 0
            else:
                delay -= self.placeholder
                self.placeholder = 0
        else:
            delay = 0

        if quality == len(manifest.bitrates) - 1:
            delay = 0

        # insufficient buffer rule
        if not self.no_ibr:
            safe_size = self.ibr_safety * (buffer_level - latency) * throughput
            self.ibr_safety *= BolaEnh.low_buffer_safety_factor_init
            self.ibr_safety = max(self.ibr_safety, BolaEnh.low_buffer_safety_factor)
            for q in range(quality):
                if manifest.bitrates[q + 1] * manifest.segment_time > safe_size:
                    #print('InsufficientBufferRule %d -> %d' % (quality, q))
                    quality = q
                    delay = 0
                    min_level = self.min_buffer_for_quality(quality)
                    max_placeholder = max(0, min_level - buffer_level)
                    self.placeholder = min(max_placeholder, self.placeholder)
                    break

        #print('ph=%d' % self.placeholder)
        return (quality, delay)

    def report_delay(self, delay):
        self.placeholder += delay

    def report_download(self, metrics, is_replacment):
        global manifest
        self.last_quality = metrics.quality
        level = self.util.get_buffer_level()

        if metrics.abandon_to_quality == None:

            if is_replacment:
                self.placeholder += manifest.segment_time
            else:
                # make sure placeholder is not too large relative to download
                level_was = level + metrics.time
                max_effective_level = self.max_buffer_for_quality(metrics.quality)
                max_placeholder = max(0, max_effective_level - level_was)
                self.placeholder = min(self.placeholder, max_placeholder)

                # make sure placeholder not too small (can happen when decision not taken by BOLA)
                if level > 0:
                    # we don't want to inflate placeholder when rebuffering
                    min_effective_level = self.min_buffer_for_quality(metrics.quality)
                    # min_effective_level < max_effective_level
                    min_placeholder = min_effective_level - level_was
                    self.placeholder = max(self.placeholder, min_placeholder)
                # else: no need to deflate placeholder for 0 buffer - empty buffer handled

        elif not is_replacment: # do nothing if we abandoned a replacement
            # abandonment indicates something went wrong - lower placeholder to conservative level
            if metrics.abandon_to_quality > 0:
                want_level = self.min_buffer_for_quality(metrics.abandon_to_quality)
            else:
                want_level = BolaEnh.minimum_buffer
            max_placeholder = max(0, want_level - level)
            self.placeholder = min(self.placeholder, max_placeholder)

    def report_seek(self, where):
        # TODO: seek properly
        self.state = BolaEnh.State.STARTUP

    def check_abandon(self, progress, buffer_level):
        global manifest

        remain = progress.size - progress.downloaded
        if progress.downloaded <= 0 or remain <= 0:
            return None

        # abandon leads to new latency, so estimate what current status is after latency
        bl = max(0, buffer_level + self.placeholder - progress.time_to_first_bit)
        tp = progress.downloaded / (progress.time - progress.time_to_first_bit)
        sz = remain - progress.time_to_first_bit * tp
        if sz <= 0:
            return None

        abandon_to = None
        score = (self.Vp * (self.gp + self.utilities[progress.quality]) - bl) / sz

        for q in range(progress.quality):
            other_size = progress.size * manifest.bitrates[q] / manifest.bitrates[progress.quality]
            other_score = (self.Vp * (self.gp + self.utilities[q]) - bl) / other_size
            if other_size < sz and other_score > score:
                # check size:
                # if remaining bits in this download are less than new download, why switch?
                # IMPORTANT: this check is NOT subsumed in score check:
                # if sz < other_size and bl is large, original score suffers larger penalty
                #print('abandon bl=%d=%d+%d-%d %d->%d score:%d->%s' % (progress.quality, bl, buffer_level, self.placeholder, progress.time_to_first_bit, q, score, other_score))
                score = other_score
                abandon_to = q

        return abandon_to

class AbrInput(Abr):

    def __init__(self, path, config):
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.abr_module = SourceFileLoader(self.name, path).load_module()
        self.abr_class = getattr(self.abr_module, self.name)
        self.abr_class.session = session_info
        self.abr = self.abr_class(config)

    def get_quality_delay(self, segment_index):
        return self.abr.get_quality_delay(segment_index)
    def get_first_quality(self):
        return self.abr.get_first_quality()
    def report_delay(self, delay):
        self.abr.report_delay(delay)
    def report_download(self, metrics, is_replacment):
        self.abr.report_download(metrics, is_replacment)
    def report_seek(self, where):
        self.abr.report_seek(where)
    def check_abandon(self, progress, buffer_level):
        return self.abr.check_abandon(progress, buffer_level)

abr_list = {}
abr_list['bola'] = Bola
abr_list['bolae'] = BolaEnh
# abr_list['bba'] = Bba
# abr_list['throughput'] = ThroughputRule
# abr_list['dynamic'] = Dynamic
# abr_list['dynamicdash'] = DynamicDash
abr_default = 'bolae'

class Replacement:

    session = session_info

    def check_replace(self, quality):
        return None
    def check_abandon(self, progress, buffer_level):
        return None

class NoReplace(Replacement):
        pass

class Replace(Replacement):

    def __init__(self, strategy):
        self.strategy = strategy
        self.replacing = None
        # self.replacing is either None or -ve index to buffer_contents

    def check_replace(self, quality):
        global manifest
        global buffer_contents
        global buffer_fcc

        self.replacing = None

        if self.strategy == 0:

            skip = math.ceil(1.5 + buffer_fcc / manifest.segment_time)
            print('skip = %d  fcc = %d' % (skip, buffer_fcc))
            for i in range(skip, len(buffer_contents)):
                if buffer_contents[i] < quality:
                    self.replacing = i - len(buffer_contents)
                    break

            if self.replacing == None:
               print('no repl:  0/%d' % len(buffer_contents))
            else:
               print('replace: %d/%d' % (self.replacing, len(buffer_contents)))

        elif self.strategy == 1:

            skip = math.ceil(1.5 + buffer_fcc / manifest.segment_time)
            print('skip = %d  fcc = %d' % (skip, buffer_fcc))
            for i in range(len(buffer_contents) - 1, skip - 1, -1):
                if buffer_contents[i] < quality:
                    self.replacing = i - len(buffer_contents)
                    break

            if self.replacing == None:
               print('no repl:  0/%d' % len(buffer_contents))
            else:
               print('replace: %d/%d' % (self.replacing, len(buffer_contents)))

        else:
            pass


        return self.replacing

    def check_abandon(self, progress, buffer_level):
        global manifest
        global buffer_contents
        global buffer_fcc

        if self.replacing == None:
            return None
        if buffer_level + manifest.segment_time * self.replacing <= 0:
            return -1
        return None

class ReplacementInput(Replacement):

    def __init__(self, path):
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.replacement_module = SourceFileLoader(self.name, path).load_module()
        self.replacement_class = getattr(self.replacement_module, self.name)
        self.replacement_class.session = session_info
        self.replacement = self.replacement_class()

    def check_replace(self, quality):
        return self.replacement.check_replace(quality)
    def check_abandon(self, progress, buffer_level):
        return self.replacement.check_abandon(progress, buffer_level)

class Args:
  def __init__(self):
    self.network = 'example/network.json'
    self.network_multiplier = 1
    self.movie = 'example/movie.json'
    self.movie_length = None
    self.abr = abr_default
    self.abr_basic = False
    self.abr_osc = False
    self.gamma_p = 5
    self.no_insufficient_buffer_rule = False
    self.moving_average = average_default
    self.window_size = [3]
    self.half_life = [3, 8]
    self.seek = None
    self.replace = 'none'
    self.max_buffer = 25
    self.no_abandon = False
    self.rampup_threshold = None
    self.verbose = False

def initSabre():
    global args, verbose, buffer_contents, buffer_fcc, pending_quality_up, reaction_metrics, rebuffer_event_count
    global rebuffer_time, played_utility, played_bitrate, total_play_time, total_bitrate_change, total_log_bitrate_change
    global total_reaction_time, last_played, overestimate_count, overestimate_average, goodestimate_count
    global goodestimate_average, estimate_average, rampup_origin, rampup_time, rampup_threshold
    global max_buffer_size, manifest, bitrates, utility_offset, utilities, network_trace, buffer_size, gamma_p
    global config, abr, network, replacer, throughput_history, quality, size, download_metric, download_time, startup_time
    global t, l, next_segment, abandoned_to_quality, full_delay, current_segment, check_abandon
    global delay, replace, throughput

    args = Args()

    verbose = args.verbose

    buffer_contents = []
    buffer_fcc = 0
    pending_quality_up = []
    reaction_metrics = []

    rebuffer_event_count = 0
    rebuffer_time = 0
    played_utility = 0
    played_bitrate = 0
    total_play_time = 0
    total_bitrate_change = 0
    total_log_bitrate_change = 0
    total_reaction_time = 0
    last_played = None

    overestimate_count = 0
    overestimate_average = 0
    goodestimate_count = 0
    goodestimate_average = 0
    estimate_average = 0

    rampup_origin = 0
    rampup_time = None
    rampup_threshold = args.rampup_threshold

    max_buffer_size = args.max_buffer * 1000

    ManifestInfo = namedtuple('ManifestInfo', 'segment_time bitrates utilities segments')
    NetworkPeriod = namedtuple('NetworkPeriod', 'time bandwidth latency')
    utils = Util()

    manifest = utils.load_json(args.movie)
    bitrates = manifest['bitrates_kbps']
    utility_offset = 0 - math.log(bitrates[0])     
    utilities = [math.log(b) + utility_offset for b in bitrates]
    if args.movie_length != None:
        l1 = len(manifest['segment_sizes_bits'])
        l2 = math.ceil(args.movie_length * 1000 / manifest['segment_duration_ms'])
        manifest['segment_sizes_bits'] *= math.ceil(l2 / l1)
        manifest['segment_sizes_bits'] = manifest['segment_sizes_bits'][0:l2]
    manifest = ManifestInfo(segment_time = manifest['segment_duration_ms'],
                            bitrates     = bitrates,
                            utilities    = utilities,
                            segments     = manifest['segment_sizes_bits'])
    SessionInfo.manifest = manifest

    network_trace = utils.load_json(args.network)
    network_trace = [NetworkPeriod(time      = p['duration_ms'],
                                   bandwidth = p['bandwidth_kbps'] * args.network_multiplier,
                                   latency   = p['latency_ms'])
                     for p in network_trace]


    buffer_size = args.max_buffer * 1000
    gamma_p = args.gamma_p

    config = {'buffer_size': buffer_size,
              'gp': gamma_p,
              'abr_osc': args.abr_osc,
              'abr_basic': args.abr_basic,
              'no_ibr': args.no_insufficient_buffer_rule}
    if args.abr[-3:] == '.py':
        abr = AbrInput(args.abr, config)
    else:
        abr_list[args.abr].use_abr_o = args.abr_osc
        abr_list[args.abr].use_abr_u = not args.abr_osc
        abr = abr_list[args.abr](config, utils)
    network = NetworkModel(network_trace, utils)

    if args.replace[-3:] == '.py':
        replacer = ReplacementInput(args.replace)
    if args.replace == 'left':
        replacer = Replace(0)
    elif args.replace == 'right':
        replacer = Replace(1)
    else:
        replacer = NoReplace()

    config = {'window_size': args.window_size, 'half_life': args.half_life}
    throughput_history = average_list[args.moving_average](config)

    quality = abr.get_first_quality()
    size = manifest.segments[0][quality]
    download_metric = network.download(size, 0, quality, 0)
    download_time = download_metric.time - download_metric.time_to_first_bit
    # Startzeit
    startup_time = download_time
    #print('Startzeit: ' + str(startup_time))
    buffer_contents.append(download_metric.quality)
    t = download_metric.size / download_time
    l = download_metric.time_to_first_bit
    throughput_history.push(download_time, t, l)
    total_play_time += download_metric.time

    if verbose:
        print('[%d-%d]  %d: q=%d s=%d/%d t=%d=%d+%d bl=0->0->%d' %
              (0, round(download_metric.time), 0, download_metric.quality,
               download_metric.downloaded, download_metric.size,
               download_metric.time, download_metric.time_to_first_bit,
               download_metric.time - download_metric.time_to_first_bit,
               utils.get_buffer_level()))

    next_segment = 1
    abandoned_to_quality = None
    while next_segment < len(manifest.segments):

        if args.seek != None:
            if next_segment * manifest.segment_time >= 1000 * args.seek[0]:
                next_segment = math.floor(1000 * args.seek[1] / manifest.segment_time)
                buffer_contents = []
                buffer_fcc = 0
                abr.report_seek(1000 * args.seek[1])
                args.seek = None
                rampup_origin = total_play_time
                rampup_time = None

        full_delay = utils.get_buffer_level() + manifest.segment_time - buffer_size
        if full_delay > 0:
            utils.deplete_buffer(full_delay)
            network.delay(full_delay)
            abr.report_delay(full_delay)
            if verbose:
                print('full buffer delay %d bl=%d' % (full_delay, utils.get_buffer_level()))

        if abandoned_to_quality == None:
            (quality, delay) = abr.get_quality_delay(next_segment)
            replace = replacer.check_replace(quality)
        else:
            (quality, delay) = (abandoned_to_quality, 0)
            replace = None
            abandon_to_quality = None

        if replace != None:
            delay = 0
            current_segment = next_segment + replace
            check_abandon = replacer.check_abandon
        else:
            current_segment = next_segment
            check_abandon = abr.check_abandon
        if args.no_abandon:
            check_abandon = None

        size = manifest.segments[current_segment][quality]

        if delay > 0:
            utils.deplete_buffer(delay)
            network.delay(delay)
            if verbose:
                print('abr delay %d bl=%d' % (delay, utils.get_buffer_level()))
        
        download_metric = network.download(size, current_segment, quality,
                                           utils.get_buffer_level(), check_abandon)

        if verbose:
            print('[%d-%d]  %d: q=%d s=%d/%d t=%d=%d+%d ' %
                  (round(total_play_time), round(total_play_time + download_metric.time),
                   current_segment, download_metric.quality,
                   download_metric.downloaded, download_metric.size,
                   download_metric.time, download_metric.time_to_first_bit,
                   download_metric.time - download_metric.time_to_first_bit),
                  end = '')
            if replace == None:
                if download_metric.abandon_to_quality == None:
                    print('bl=%d' % utils.get_buffer_level(), end = '')
                else:
                    print(' ABANDONED to %d - %d/%d bits in %d=%d+%d ttfb+ttdl  bl=%d' %
                          (download_metric.abandon_to_quality,
                           download_metric.downloaded, download_metric.size,
                           download_metric.time, download_metric.time_to_first_bit,
                           download_metric.time - download_metric.time_to_first_bit,
                           utils.get_buffer_level()),
                          end = '')
            else:
                if download_metric.abandon_to_quality == None:
                    print(' REPLACEMENT  bl=%d' % utils.get_buffer_level(), end = '')
                else:
                    print(' REPLACMENT ABANDONED after %d=%d+%d ttfb+ttdl  bl=%d' %
                          (download_metric.time, download_metric.time_to_first_bit,
                           download_metric.time - download_metric.time_to_first_bit,
                           utils.get_buffer_level()),
                          end = '')

        utils.deplete_buffer(download_metric.time)
        if verbose:
            print('->%d' % utils.get_buffer_level(), end='')

        if replace == None:
            if download_metric.abandon_to_quality == None:
                buffer_contents += [quality]
                next_segment += 1
            else:
                abandon_to_quality = download_metric.abandon_to_quality
        else:
            if download_metric.abandon_to_quality == None:
                if utils.get_buffer_level() + manifest.segment_time * replace >= 0:
                    buffer_contents[replace] = quality
                else:
                    print('WARNING: too late to replace')
                    pass
            else:
                pass

        if verbose:
            print('->%d' % utils.get_buffer_level())

        abr.report_download(download_metric, replace != None)

        download_time = download_metric.time - download_metric.time_to_first_bit
        t = download_metric.downloaded / download_time
        l = download_metric.time_to_first_bit

        if throughput > t:
            overestimate_count += 1
            overestimate_average += (throughput - t - overestimate_average) / overestimate_count
        else:
            goodestimate_count += 1
            goodestimate_average += (t - throughput - goodestimate_average) / goodestimate_count
        estimate_average += ((throughput - t - estimate_average) /
                             (overestimate_count + goodestimate_count))

        if download_metric.abandon_to_quality == None:
            throughput_history.push(download_time, t, l)

    utils.playout_buffer()

    to_time_average = 1 / (total_play_time / manifest.segment_time)
    count = len(manifest.segments)
    time = count * manifest.segment_time + rebuffer_time + startup_time
    
    result = {}

    result['buffer_size'] = buffer_size
    result['total_played_utility'] = played_utility
    result['time_average_played_utility'] = played_utility * to_time_average
    result['total_played_bitrate'] = played_bitrate
    result['time_average_played_bitrate'] = played_bitrate * to_time_average
    result['total_play_time'] = total_play_time / 1000
    result['total_play_time_chunks'] = total_play_time / manifest.segment_time
    result['total_rebuffer'] = rebuffer_time / 1000
    result['rebuffer_ratio'] = rebuffer_time / total_play_time
    result['time_average_rebuffer'] = rebuffer_time / 1000 * to_time_average
    result['total_rebuffer_events'] = rebuffer_event_count
    result['time_average_rebuffer_events'] = rebuffer_event_count * to_time_average
    result['total_bitrate_change'] = total_bitrate_change
    result['time_average_bitrate_change'] = total_bitrate_change * to_time_average
    result['total_log_bitrate_change'] = total_log_bitrate_change
    result['time_average_log_bitrate_change'] = total_log_bitrate_change * to_time_average
    result['time_average_score'] = to_time_average * (played_utility - args.gamma_p * rebuffer_time / manifest.segment_time)

    if overestimate_count == 0:
        result['over_estimate_count'] = 0
        result['over_estimate'] = 0
    else:
        result['over_estimate_count'] = overestimate_count
        result['over_estimate'] = overestimate_average

    if goodestimate_count == 0:
        result['leq_estimate_count'] = 0
        result['leq_estimate'] = 0
    else:
        result['leq_estimate_count'] = goodestimate_count
        result['leq_estimate'] = goodestimate_average

    result['estimate'] = estimate_average

    if rampup_time == None:
        result['rampup_time'] = len(manifest.segments) * manifest.segment_time / 1000
    else:
        result['rampup_time'] = rampup_time / 1000

    result['total_reaction_time'] = total_reaction_time / 1000

    return result

if __name__ == '__main__':
    result = initSabre()
    
    for key, value in result.items():
        print(f'{key}: {value}')