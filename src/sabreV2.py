import json
import math
import os
from importlib.machinery import SourceFileLoader
from collections import namedtuple
from enum import Enum

# Units used throughout:
#     size     : bits
#     time     : ms
#     size/time: bits/ms = kbit/s


# global variables:
#     video manifest
#     buffer contents
#     buffer first segment consumed
#     throughput estimate
#     latency estimate
#     rebuffer event count
#     rebuffer total time
#     session info

class Util:

    def __init__(self,
                 buffer_contents=[],
                 buffer_fcc=0,
                 last_played=None,
                 latency=None,
                 manifest=None,
                 max_buffer_size=None,
                 network_total_time=None,
                 pending_quality_up=[],
                 played_bitrate=0,
                 played_utility=0,
                 rampup_origin=0,
                 rampup_threshold=None,
                 rampup_time=None,
                 rebuffer_event_count=0,
                 rebuffer_time=0,
                 sustainable_quality=None,
                 throughput=0,
                 total_bitrate_change=0,
                 total_log_bitrate_change=0,
                 total_play_time=0,
                 total_reaction_time=0,
                 verbose=True):
        self.buffer_contents = buffer_contents
        self.buffer_fcc = buffer_fcc
        self.last_played = last_played
        self.latency = latency
        self.manifest = manifest
        self.max_buffer_size = max_buffer_size
        self.network_total_time = network_total_time
        self.pending_quality_up = pending_quality_up
        self.played_bitrate = played_bitrate
        self.played_utility = played_utility
        self.rampup_origin = rampup_origin
        self.rampup_threshold = rampup_threshold
        self.rampup_time = rampup_time
        self.rebuffer_event_count = rebuffer_event_count
        self.rebuffer_time = rebuffer_time
        self.sustainable_quality = sustainable_quality
        self.throughput = throughput
        self.total_bitrate_change = total_bitrate_change
        self.total_log_bitrate_change = total_log_bitrate_change
        self.total_play_time = total_play_time
        self.total_reaction_time = total_reaction_time
        self.verbose = verbose
        self.session_info = SessionInfo(self)

    def load_json(self, path):
        with open(path) as file:
            obj = json.load(file)
        return obj

    def get_buffer_level(self):
        return self.manifest.segment_time * len(self.buffer_contents) - self.buffer_fcc

    def deplete_buffer(self, time):

        if len(self.buffer_contents) == 0:
            self.rebuffer_time += time
            self.total_play_time += time
            return

        if self.buffer_fcc > 0:
            # first play any partial chunk left

            if time + self.buffer_fcc < self.manifest.segment_time:
                self.buffer_fcc += time
                self.total_play_time += time
                return

            time -= self.manifest.segment_time - self.buffer_fcc
            self.total_play_time += self.manifest.segment_time - self.buffer_fcc
            self.buffer_contents.pop(0)
            self.buffer_fcc = 0

        # buffer_fcc == 0 if we're here

        while time > 0 and len(self.buffer_contents) > 0:

            quality = self.buffer_contents[0]
            self.played_utility += self.manifest.utilities[quality]
            self.played_bitrate += self.manifest.bitrates[quality]
            if quality != self.last_played and self.last_played != None:
                self.total_bitrate_change += abs(self.manifest.bitrates[quality] -
                                                 self.manifest.bitrates[self.last_played])
                self.total_log_bitrate_change += abs(math.log(self.manifest.bitrates[quality] /
                                                              self.manifest.bitrates[self.last_played]))
            self.last_played = quality

            if self.rampup_time == None:
                rt = self.sustainable_quality if self.rampup_threshold == None else self.rampup_threshold
                if quality >= rt:
                    self.rampup_time = self.total_play_time - self.rampup_origin

            # bookkeeping to track reaction time to increased bandwidth
            for p in self.pending_quality_up:
                if len(p) == 2 and quality >= p[1]:
                    p.append(self.total_play_time)

            if time >= self.manifest.segment_time:
                self.buffer_contents.pop(0)
                self.total_play_time += self.manifest.segment_time
                time -= self.manifest.segment_time
            else:
                self.buffer_fcc = time
                self.total_play_time += time
                time = 0

        if time > 0:
            self.rebuffer_time += time
            self.total_play_time += time
            self.rebuffer_event_count += 1

        self.process_quality_up(self.total_play_time)

    def playout_buffer(self):
        self.deplete_buffer(self.get_buffer_level())

        # make sure no rounding error
        del self.buffer_contents[:]
        self.buffer_fcc = 0

    def process_quality_up(self, now):
        # check which switches can be processed

        cutoff = now - self.max_buffer_size

        while len(self.pending_quality_up) > 0 and self.pending_quality_up[0][0] < cutoff:
            p = self.pending_quality_up.pop(0)
            if len(p) == 2:
                reaction = self.max_buffer_size
            else:
                reaction = min(self.max_buffer_size, p[2] - p[0])
            # print('\n[%d] reaction time: %d' % (now, reaction))
            # print(p)
            self.total_reaction_time += reaction

    def advertize_new_network_quality(self, quality, previous_quality):
        # bookkeeping to track reaction time to increased bandwidth

        # process any previous quality up switches that have "matured"
        self.process_quality_up(self.network_total_time)

        # mark any pending switch up done if new quality switches back below its quality
        for p in self.pending_quality_up:
            if len(p) == 2 and p[1] > quality:
                p.append(self.network_total_time)
        # pending_quality_up = [p for p in pending_quality_up if p[1] >= quality]

        # filter out switches which are not upwards (three separate checks)
        if quality <= previous_quality:
            return
        for q in self.buffer_contents:
            if quality <= q:
                return
        for p in self.pending_quality_up:
            if quality <= p[1]:
                return

        # valid quality up switch
        # print([network_total_time, quality])
        self.pending_quality_up.append([self.network_total_time, quality])


class NetworkModel:

    DownloadProgress = namedtuple('DownloadProgress',
                                  'index quality '
                                  'size downloaded '
                                  'time time_to_first_bit '
                                  'abandon_to_quality')

    min_progress_size = 12000
    min_progress_time = 50

    def __init__(self, network_trace, util):
        self.util = util

        self.util.sustainable_quality = None
        self.util.network_total_time = 0
        self.trace = network_trace
        self.index = -1
        self.time_to_next = 0
        self.next_network_period()

    def next_network_period(self):
        self.index += 1
        if self.index == len(self.trace):
            self.index = 0
        self.time_to_next = self.trace[self.index].time

        latency_factor = 1 - \
            self.trace[self.index].latency / self.util.manifest.segment_time
        effective_bandwidth = self.trace[self.index].bandwidth * latency_factor

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

        if self.util.verbose:
            print('[%d] Network: %d,%d  (q=%d: bitrate=%d)' %
                  (round(self.util.network_total_time),
                   self.trace[self.index].bandwidth, self.trace[self.index].latency,
                   self.util.sustainable_quality, self.util.manifest.bitrates[self.util.sustainable_quality]))

    # return delay time
    def do_latency_delay(self, delay_units):
        total_delay = 0
        while delay_units > 0:
            current_latency = self.trace[self.index].latency
            time = delay_units * current_latency
            if time <= self.time_to_next:
                total_delay += time
                self.util.network_total_time += time
                self.time_to_next -= time
                delay_units = 0
            else:
                # time > self.time_to_next implies current_latency > 0
                total_delay += self.time_to_next
                self.util.network_total_time += self.time_to_next
                delay_units -= self.time_to_next / current_latency
                self.next_network_period()
        return total_delay

    # return download time
    def do_download(self, size):
        total_download_time = 0
        while size > 0:
            current_bandwidth = self.trace[self.index].bandwidth
            if size <= self.time_to_next * current_bandwidth:
                # current_bandwidth > 0
                time = size / current_bandwidth
                total_download_time += time
                self.util.network_total_time += time
                self.time_to_next -= time
                size = 0
            else:
                total_download_time += self.time_to_next
                self.util.network_total_time += self.time_to_next
                size -= self.time_to_next * current_bandwidth
                self.next_network_period()
        return total_download_time

    def do_minimal_latency_delay(self, delay_units, min_time):
        total_delay_units = 0
        total_delay_time = 0
        while delay_units > 0 and min_time > 0:
            current_latency = self.trace[self.index].latency
            time = delay_units * current_latency
            if time <= min_time and time <= self.time_to_next:
                units = delay_units
                self.time_to_next -= time
                self.util.network_total_time += time
            elif min_time <= self.time_to_next:
                # time > 0 implies current_latency > 0
                time = min_time
                units = time / current_latency
                self.time_to_next -= time
                self.util.network_total_time += time
            else:
                time = self.time_to_next
                units = time / current_latency
                self.util.network_total_time += time
                self.next_network_period()
            total_delay_units += units
            total_delay_time += time
            delay_units -= units
            min_time -= time
        return (total_delay_units, total_delay_time)

    def do_minimal_download(self, size, min_size, min_time):
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
                    self.util.network_total_time += time
                elif min_bits <= bits_to_next:
                    bits = min_bits
                    time = bits / current_bandwidth
                    # make sure rounding error does not push while loop into endless loop
                    min_size = 0
                    min_time = 0
                    self.time_to_next -= time
                    self.util.network_total_time += time
                else:
                    bits = bits_to_next
                    time = self.time_to_next
                    self.util.network_total_time += time
                    self.next_network_period()
            else:  # current_bandwidth == 0
                bits = 0
                if min_size > 0 or min_time > self.time_to_next:
                    time = self.time_to_next
                    self.util.network_total_time += time
                    self.next_network_period()
                else:
                    time = min_time
                    self.time_to_next -= time
                    self.util.network_total_time += time
            total_size += bits
            total_time += time
            size -= bits
            min_size -= bits
            min_time -= time
        return (total_size, total_time)

    def delay(self, time):
        while time > self.time_to_next:
            time -= self.time_to_next
            self.util.network_total_time += self.time_to_next
            self.next_network_period()
        self.time_to_next -= time
        self.util.network_total_time += time

    def download(self, size, idx, quality, buffer_level, check_abandon=None):
        if size <= 0:
            return self.DownloadProgress(index=idx, quality=quality,
                                         size=0, downloaded=0,
                                         time=0, time_to_first_bit=0,
                                         abandon_to_quality=None)

        if not check_abandon or (NetworkModel.min_progress_time <= 0 and
                                 NetworkModel.min_progress_size <= 0):
            latency = self.do_latency_delay(1)
            time = latency + self.do_download(size)
            return self.DownloadProgress(index=idx, quality=quality,
                                         size=size, downloaded=size,
                                         time=time, time_to_first_bit=latency,
                                         abandon_to_quality=None)

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
                (units, time) = self.do_minimal_latency_delay(
                    delay_units, min_time_to_progress)
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

        return self.DownloadProgress(index=idx, quality=quality,
                                     size=size, downloaded=total_download_size,
                                     time=total_download_time, time_to_first_bit=latency,
                                     abandon_to_quality=abandon_quality)


class ThroughputHistory:
    def __init__(self, config):
        pass

    def push(self, time, tput, lat):
        raise NotImplementedError


class SessionInfo:

    def __init__(self, util):
        self.util = util

    def get_throughput(self):
        return self.util.throughput

    def get_buffer_contents(self):
        return self.util.buffer_contents[:]


class Abr:

    def __init__(self, config, util):
        self.util = util
        self.session = util.session_info

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
        p = self.util.manifest.segment_time
        quality = 0
        while (quality + 1 < len(self.util.manifest.bitrates) and
               self.util.latency + p * self.util.manifest.bitrates[quality + 1] / tput <= p):
            quality += 1
        return quality


class Replacement:

    def __init__(self, util) -> None:
        self.util = util
        self.session = util.session_info

    def check_replace(self, quality):
        return None

    def check_abandon(self, progress, buffer_level):
        return None


class SlidingWindow(ThroughputHistory):

    default_window_size = [3]
    max_store = 20

    def __init__(self, config, util):
        self.util = util

        if 'window_size' in config and config['window_size'] != None:
            self.window_size = config['window_size']
        else:
            self.window_size = SlidingWindow.default_window_size

        # TODO: init somewhere else?
        self.util.throughput = None
        self.util.latency = None

        self.last_throughputs = []
        self.last_latencies = []

    def push(self, time, tput, lat):
        self.last_throughputs += [tput]
        self.last_throughputs = self.last_throughputs[-SlidingWindow.max_store:]

        self.last_latencies += [lat]
        self.last_latencies = self.last_latencies[-SlidingWindow.max_store:]

        tput = None
        lat = None
        for ws in self.window_size:
            sample = self.last_throughputs[-ws:]
            t = sum(sample) / len(sample)
            tput = t if tput == None else min(tput, t)  # conservative min
            sample = self.last_latencies[-ws:]
            l = sum(sample) / len(sample)
            lat = l if lat == None else max(lat, l)  # conservative max
        self.util.throughput = tput
        self.util.latency = lat


class Ewma(ThroughputHistory):

    # for throughput:
    default_half_life = [8000, 3000]

    def __init__(self, config, util):
        self.util = util

        # TODO: init somewhere else?
        self.util.throughput = None
        self.util.latency = None

        if 'half_life' in config and config['half_life'] != None:
            self.half_life = [h * 1000 for h in config['half_life']]
        else:
            self.half_life = Ewma.default_half_life

        self.latency_half_life = [
            h / self.util.manifest.segment_time for h in self.half_life]

        self.throughput = [0] * len(self.half_life)
        self.weight_throughput = 0
        self.latency = [0] * len(self.half_life)
        self.weight_latency = 0

    def push(self, time, tput, lat):

        for i in range(len(self.half_life)):
            alpha = math.pow(0.5, time / self.half_life[i])
            self.throughput[i] = alpha * \
                self.throughput[i] + (1 - alpha) * tput
            alpha = math.pow(0.5, 1 / self.latency_half_life[i])
            self.latency[i] = alpha * self.latency[i] + (1 - alpha) * lat

        self.weight_throughput += time
        self.weight_latency += 1

        tput = None
        lat = None
        for i in range(len(self.half_life)):
            zero_factor = 1 - \
                math.pow(0.5, self.weight_throughput / self.half_life[i])
            t = self.throughput[i] / zero_factor
            tput = t if tput == None else min(
                tput, t)  # conservative case is min
            zero_factor = 1 - \
                math.pow(0.5, self.weight_latency / self.latency_half_life[i])
            l = self.latency[i] / zero_factor
            lat = l if lat == None else max(lat, l)  # conservative case is max
        self.util.throughput = tput
        self.util.latency = lat


class Bola(Abr):

    def __init__(self, config, util):
        self.util = util

        # so utilities[0] = 0
        utility_offset = -math.log(self.util.manifest.bitrates[0])
        self.utilities = [
            math.log(b) + utility_offset for b in self.util.manifest.bitrates]

        self.gp = config['gp']
        self.buffer_size = config['buffer_size']
        self.abr_osc = config['abr_osc']
        self.abr_basic = config['abr_basic']
        self.Vp = (self.buffer_size - self.util.manifest.segment_time) / \
            (self.utilities[-1] + self.gp)

        self.last_seek_index = 0  # TODO
        self.last_quality = 0

        if self.util.verbose:
            for q in range(len(self.util.manifest.bitrates)):
                b = self.util.manifest.bitrates[q]
                u = self.utilities[q]
                l = self.Vp * (self.gp + u)
                if q == 0:
                    print('%d %d' % (q, l))
                else:
                    qq = q - 1
                    bb = self.util.manifest.bitrates[qq]
                    uu = self.utilities[qq]
                    ll = self.Vp * (self.gp + (b * uu - bb * u) / (b - bb))
                    print('%d %d    <- %d %d' % (q, l, qq, ll))

    def quality_from_buffer(self):
        level = self.util.get_buffer_level()
        quality = 0
        score = None
        for q in range(len(self.util.manifest.bitrates)):
            s = (
                (self.Vp * (self.utilities[q] + self.gp) - level) / self.util.manifest.bitrates[q])
            if score == None or s > score:
                quality = q
                score = s
        return quality

    def get_quality_delay(self, segment_index):

        if not self.abr_basic:
            t = min(segment_index - self.last_seek_index,
                    len(self.util.manifest.segments) - segment_index)
            t = max(t / 2, 3)
            t = t * self.util.manifest.segment_time
            buffer_size = min(self.buffer_size, t)
            self.Vp = (buffer_size - self.util.manifest.segment_time) / \
                (self.utilities[-1] + self.gp)

        quality = self.quality_from_buffer()
        delay = 0

        if quality > self.last_quality:
            quality_t = self.quality_from_throughput(self.util.throughput)
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
                    b = self.util.manifest.bitrates[quality]
                    u = self.utilities[quality]
                    # bb = manifest.bitrates[quality + 1]
                    # uu = self.utilities[quality + 1]
                    # l = self.Vp * (self.gp + (bb * u - b * uu) / (bb - b))
                    l = self.Vp * (self.gp + u)
                    delay = max(0, self.util.get_buffer_level() - l)
                    if quality == len(self.util.manifest.bitrates) - 1:
                        delay = 0
                    # delay = 0 ###########

        self.last_quality = quality
        return (quality, delay)

    def report_seek(self, where):
        # TODO: seek properly
        self.last_seek_index = math.floor(
            where / self.util.manifest.segment_time)

    def check_abandon(self, progress, buffer_level):
        # global manifest

        if self.abr_basic:
            return None

        remain = progress.size - progress.downloaded
        if progress.downloaded <= 0 or remain <= 0:
            return None

        abandon_to = None
        score = (
            self.Vp * (self.gp + self.utilities[progress.quality]) - buffer_level) / remain
        if score < 0:
            return  # TODO: check

        for q in range(progress.quality):
            other_size = progress.size * \
                self.util.manifest.bitrates[q] / \
                self.util.manifest.bitrates[progress.quality]
            other_score = (
                self.Vp * (self.gp + self.utilities[q]) - buffer_level) / other_size
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
        self.util = util

        config_buffer_size = config['buffer_size']
        self.abr_osc = config['abr_osc']
        self.no_ibr = config['no_ibr']

        # so utilities[0] = 1
        utility_offset = 1 - math.log(self.util.manifest.bitrates[0])
        self.utilities = [
            math.log(b) + utility_offset for b in self.util.manifest.bitrates]

        if self.no_ibr:
            self.gp = config['gp'] - 1  # to match BOLA Basic
            buffer = config['buffer_size']
            self.Vp = (buffer - self.util.manifest.segment_time) / \
                (self.utilities[-1] + self.gp)
        else:
            buffer = BolaEnh.minimum_buffer
            buffer += BolaEnh.minimum_buffer_per_level * \
                len(self.util.manifest.bitrates)
            buffer = max(buffer, config_buffer_size)
            self.gp = (self.utilities[-1] - 1) / \
                (buffer / BolaEnh.minimum_buffer - 1)
            self.Vp = BolaEnh.minimum_buffer / self.gp
            # equivalently:
            # self.Vp = (buffer - BolaEnh.minimum_buffer) / (math.log(manifest.bitrates[-1] / manifest.bitrates[0]))
            # self.gp = BolaEnh.minimum_buffer / self.Vp

        self.state = BolaEnh.State.STARTUP
        self.placeholder = 0
        self.last_quality = 0

        if self.util.verbose:
            for q in range(len(self.util.manifest.bitrates)):
                b = self.util.manifest.bitrates[q]
                u = self.utilities[q]
                l = self.Vp * (self.gp + u)
                if q == 0:
                    print('%d %d' % (q, l))
                else:
                    qq = q - 1
                    bb = self.util.manifest.bitrates[qq]
                    uu = self.utilities[qq]
                    ll = self.Vp * (self.gp + (b * uu - bb * u) / (b - bb))
                    print('%d %d    <- %d %d' % (q, l, qq, ll))

    def quality_from_buffer(self, level):
        if level == None:
            level = self.util.get_buffer_level()
        quality = 0
        score = None
        for q in range(len(self.util.manifest.bitrates)):
            s = (
                (self.Vp * (self.utilities[q] + self.gp) - level) / self.util.manifest.bitrates[q])
            if score == None or s > score:
                quality = q
                score = s
        return quality

    def quality_from_buffer_placeholder(self):
        return self.quality_from_buffer(self.util.get_buffer_level() + self.placeholder)

    def min_buffer_for_quality(self, quality):
        # global manifest

        bitrate = self.util.manifest.bitrates[quality]
        utility = self.utilities[quality]

        level = 0
        for q in range(quality):
            # for each bitrates[q] less than bitrates[quality],
            # BOLA should prefer bitrates[quality]
            # (unless bitrates[q] has higher utility)
            if self.utilities[q] < self.utilities[quality]:
                b = self.util.manifest.bitrates[q]
                u = self.utilities[q]
                l = self.Vp * (self.gp + (bitrate * u -
                               b * utility) / (bitrate - b))
                level = max(level, l)
        return level

    def max_buffer_for_quality(self, quality):
        return self.Vp * (self.utilities[quality] + self.gp)

    def get_quality_delay(self, segment_index):
        # global buffer_contents
        # global buffer_fcc
        # global throughput

        buffer_level = self.util.get_buffer_level()

        if self.state == BolaEnh.State.STARTUP:
            if self.util.throughput == None:
                return (self.last_quality, 0)
            self.state = BolaEnh.State.STEADY
            self.ibr_safety = BolaEnh.low_buffer_safety_factor_init
            quality = self.quality_from_throughput(self.util.throughput)
            self.placeholder = self.min_buffer_for_quality(
                quality) - buffer_level
            self.placeholder = max(0, self.placeholder)
            return (quality, 0)

        quality = self.quality_from_buffer_placeholder()
        quality_t = self.quality_from_throughput(self.util.throughput)
        if quality > self.last_quality and quality > quality_t:
            quality = max(self.last_quality, quality_t)
            if not self.abr_osc:
                quality += 1

        max_level = self.max_buffer_for_quality(quality)

        ################
        if quality > 0:
            q = quality
            b = self.util.manifest.bitrates[q]
            u = self.utilities[q]
            qq = q - 1
            bb = self.util.manifest.bitrates[qq]
            uu = self.utilities[qq]
            # max_level = self.Vp * (self.gp + (b * uu - bb * u) / (b - bb))
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

        if quality == len(self.util.manifest.bitrates) - 1:
            delay = 0

        # insufficient buffer rule
        if not self.no_ibr:
            safe_size = self.ibr_safety * \
                (buffer_level - self.util.latency) * self.util.throughput
            self.ibr_safety *= BolaEnh.low_buffer_safety_factor_init
            self.ibr_safety = max(
                self.ibr_safety, BolaEnh.low_buffer_safety_factor)
            for q in range(quality):
                if self.util.manifest.bitrates[q + 1] * self.util.manifest.segment_time > safe_size:
                    # print('InsufficientBufferRule %d -> %d' % (quality, q))
                    quality = q
                    delay = 0
                    min_level = self.min_buffer_for_quality(quality)
                    max_placeholder = max(0, min_level - buffer_level)
                    self.placeholder = min(max_placeholder, self.placeholder)
                    break

        # print('ph=%d' % self.placeholder)
        return (quality, delay)

    def report_delay(self, delay):
        self.placeholder += delay

    def report_download(self, metrics, is_replacment):
        # global manifest
        self.last_quality = metrics.quality
        level = self.util.get_buffer_level()

        if metrics.abandon_to_quality == None:

            if is_replacment:
                self.placeholder += self.util.manifest.segment_time
            else:
                # make sure placeholder is not too large relative to download
                level_was = level + metrics.time
                max_effective_level = self.max_buffer_for_quality(
                    metrics.quality)
                max_placeholder = max(0, max_effective_level - level_was)
                self.placeholder = min(self.placeholder, max_placeholder)

                # make sure placeholder not too small (can happen when decision not taken by BOLA)
                if level > 0:
                    # we don't want to inflate placeholder when rebuffering
                    min_effective_level = self.min_buffer_for_quality(
                        metrics.quality)
                    # min_effective_level < max_effective_level
                    min_placeholder = min_effective_level - level_was
                    self.placeholder = max(self.placeholder, min_placeholder)
                # else: no need to deflate placeholder for 0 buffer - empty buffer handled

        elif not is_replacment:  # do nothing if we abandoned a replacement
            # abandonment indicates something went wrong - lower placeholder to conservative level
            if metrics.abandon_to_quality > 0:
                want_level = self.min_buffer_for_quality(
                    metrics.abandon_to_quality)
            else:
                want_level = BolaEnh.minimum_buffer
            max_placeholder = max(0, want_level - level)
            self.placeholder = min(self.placeholder, max_placeholder)

    def report_seek(self, where):
        # TODO: seek properly
        self.state = BolaEnh.State.STARTUP

    def check_abandon(self, progress, buffer_level):
        # global manifest

        remain = progress.size - progress.downloaded
        if progress.downloaded <= 0 or remain <= 0:
            return None

        # abandon leads to new latency, so estimate what current status is after latency
        bl = max(0, buffer_level + self.placeholder -
                 progress.time_to_first_bit)
        tp = progress.downloaded / (progress.time - progress.time_to_first_bit)
        sz = remain - progress.time_to_first_bit * tp
        if sz <= 0:
            return None

        abandon_to = None
        score = (
            self.Vp * (self.gp + self.utilities[progress.quality]) - bl) / sz

        for q in range(progress.quality):
            other_size = progress.size * \
                self.util.manifest.bitrates[q] / self.util.manifest.bitrates[progress.quality]
            other_score = (
                self.Vp * (self.gp + self.utilities[q]) - bl) / other_size
            if other_size < sz and other_score > score:
                # check size:
                # if remaining bits in this download are less than new download, why switch?
                # IMPORTANT: this check is NOT subsumed in score check:
                # if sz < other_size and bl is large, original score suffers larger penalty
                # print('abandon bl=%d=%d+%d-%d %d->%d score:%d->%s' % (progress.quality, bl, buffer_level, self.placeholder, progress.time_to_first_bit, q, score, other_score))
                score = other_score
                abandon_to = q

        return abandon_to


class ThroughputRule(Abr):

    safety_factor = 0.9
    low_buffer_safety_factor = 0.5
    low_buffer_safety_factor_init = 0.9
    abandon_multiplier = 1.8
    abandon_grace_time = 500

    def __init__(self, config, util):
        self.ibr_safety = ThroughputRule.low_buffer_safety_factor_init
        self.no_ibr = config['no_ibr']
        self.util = util

    def get_quality_delay(self, segment_index):

        quality = self.quality_from_throughput(
            self.util.throughput * ThroughputRule.safety_factor)

        if not self.no_ibr:
            # insufficient buffer rule
            safe_size = self.ibr_safety * \
                (self.util.get_buffer_level() - self.util.latency) * self.util.throughput
            self.ibr_safety *= ThroughputRule.low_buffer_safety_factor_init
            self.ibr_safety = max(
                self.ibr_safety, ThroughputRule.low_buffer_safety_factor)
            for q in range(quality):
                if self.util.manifest.bitrates[q + 1] * self.util.manifest.segment_time > safe_size:
                    quality = q
                    break

        return (quality, 0)

    def check_abandon(self, progress, buffer_level):

        quality = None  # no abandon

        dl_time = progress.time - progress.time_to_first_bit
        if progress.time >= ThroughputRule.abandon_grace_time and dl_time > 0:
            tput = progress.downloaded / dl_time
            size_left = progress.size - progress.downloaded
            estimate_time_left = size_left / tput
            if (progress.time + estimate_time_left >
                    ThroughputRule.abandon_multiplier * self.util.manifest.segment_time):
                quality = self.quality_from_throughput(
                    tput * ThroughputRule.safety_factor)
                estimate_size = (progress.size *
                                 self.util.manifest.bitrates[quality] / self.util.manifest.bitrates[progress.quality])
                if quality >= progress.quality or estimate_size >= size_left:
                    quality = None

        return quality


class Dynamic(Abr):

    low_buffer_threshold = 10000

    def __init__(self, config, util):
        self.util = util

        self.bola = Bola(config, util)
        self.tput = ThroughputRule(config, util)

        self.is_bola = False

    def get_quality_delay(self, segment_index):
        level = self.util.get_buffer_level()

        b = self.bola.get_quality_delay(segment_index)
        t = self.tput.get_quality_delay(segment_index)

        if self.is_bola:
            if level < Dynamic.low_buffer_threshold and b[0] < t[0]:
                self.is_bola = False
        else:
            if level > Dynamic.low_buffer_threshold and b[0] >= t[0]:
                self.is_bola = True

        return b if self.is_bola else t

    def get_first_quality(self):
        if self.is_bola:
            return self.bola.get_first_quality()
        else:
            return self.tput.get_first_quality()

    def report_delay(self, delay):
        self.bola.report_delay(delay)
        self.tput.report_delay(delay)

    def report_download(self, metrics, is_replacment):
        self.bola.report_download(metrics, is_replacment)
        self.tput.report_download(metrics, is_replacment)
        if is_replacment:
            self.is_bola = False

    def check_abandon(self, progress, buffer_level):
        if False and self.is_bola:
            return self.bola.check_abandon(progress, buffer_level)
        else:
            return self.tput.check_abandon(progress, buffer_level)


class DynamicDash(Abr):

    def __init__(self, config, util):
        self.util = util

        self.bola = BolaEnh(config, util)
        self.tput = ThroughputRule(config, util)

        buffer_size = config['buffer_size']
        self.low_threshold = (buffer_size - self.util.manifest.segment_time) / 2
        self.high_threshold = (buffer_size - self.util.manifest.segment_time) - 100
        self.low_threshold = 5000
        self.high_threshold = 10000
        # TODO
        self.is_bola = False

    def get_quality_delay(self, segment_index):
        level = self.util.get_buffer_level()
        if self.is_bola and level < self.low_threshold:
            self.is_bola = False
        elif not self.is_bola and level > self.high_threshold:
            self.is_bola = True

        if self.is_bola:
            return self.bola.get_quality_delay(segment_index)
        else:
            return self.tput.get_quality_delay(segment_index)

    def get_first_quality(self):
        if self.is_bola:
            return self.bola.get_first_quality()
        else:
            return self.tput.get_first_quality()

    def report_delay(self, delay):
        self.bola.report_delay(delay)
        self.tput.report_delay(delay)

    def report_download(self, metrics, is_replacment):
        self.bola.report_download(metrics, is_replacment)
        self.tput.report_download(metrics, is_replacment)

    def check_abandon(self, progress, buffer_level):
        if self.is_bola:
            return self.bola.check_abandon(progress, buffer_level)
        else:
            return self.tput.check_abandon(progress, buffer_level)


class Bba(Abr):

    def __init__(self, config,):
        pass

    def get_quality_delay(self, segment_index):
        raise NotImplementedError

    def report_delay(self, delay):
        pass

    def report_download(self, metrics, is_replacment):
        pass

    def report_seek(self, where):
        pass


class NoReplace(Replacement):
    pass

# TODO: different classes instead of strategy


class Replace(Replacement):

    def __init__(self, strategy, util):
        self.strategy = strategy
        self.replacing = None
        self.util = util
        # self.replacing is either None or -ve index to buffer_contents

    def check_replace(self, quality):
        # global manifest
        # global buffer_contents
        # global buffer_fcc

        self.replacing = None

        if self.strategy == 0:

            skip = math.ceil(1.5 + self.util.buffer_fcc / self.util.manifest.segment_time)
            # print('skip = %d  fcc = %d' % (skip, buffer_fcc))
            for i in range(skip, len(self.util.buffer_contents)):
                if self.util.buffer_contents[i] < quality:
                    self.replacing = i - len(self.util.buffer_contents)
                    break

            # if self.replacing == None:
            #    print('no repl:  0/%d' % len(buffer_contents))
            # else:
            #    print('replace: %d/%d' % (self.replacing, len(buffer_contents)))

        elif self.strategy == 1:

            skip = math.ceil(1.5 + self.util.buffer_fcc / self.util.manifest.segment_time)
            # print('skip = %d  fcc = %d' % (skip, buffer_fcc))
            for i in range(len(self.util.buffer_contents) - 1, skip - 1, -1):
                if self.util.buffer_contents[i] < quality:
                    self.replacing = i - len(self.util.buffer_contents)
                    break

            # if self.replacing == None:
            #    print('no repl:  0/%d' % len(buffer_contents))
            # else:
            #    print('replace: %d/%d' % (self.replacing, len(buffer_contents)))

        else:
            pass

        return self.replacing

    def check_abandon(self, progress, buffer_level):
        # global manifest
        # global buffer_contents
        # global buffer_fcc

        if self.replacing == None:
            return None
        if buffer_level + self.util.manifest.segment_time * self.replacing <= 0:
            return -1
        return None


class AbrInput(Abr):

    def __init__(self, path, config):
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.abr_module = SourceFileLoader(self.name, path).load_module()
        self.abr_class = getattr(self.abr_module, self.name)
        self.abr_class.session = self.util.session_info
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


class ReplacementInput(Replacement):

    def __init__(self, path):
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.replacement_module = SourceFileLoader(
            self.name, path).load_module()
        self.replacement_class = getattr(self.replacement_module, self.name)
        self.replacement_class.session = self.util.session_info
        self.replacement = self.replacement_class()

    def check_replace(self, quality):
        return self.replacement.check_replace(quality)

    def check_abandon(self, progress, buffer_level):
        return self.replacement.check_abandon(progress, buffer_level)


ManifestInfo = namedtuple(
    'ManifestInfo', 'segment_time bitrates utilities segments')
NetworkPeriod = namedtuple('NetworkPeriod', 'time bandwidth latency')

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


def init(
    abr=abr_default,
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

    util = Util()

    util.verbose = verbose

    util.buffer_contents = []
    util.buffer_fcc = 0
    util.pending_quality_up = []

    util.rebuffer_event_count = 0
    util.rebuffer_time = 0

    util.played_utility = 0
    util.played_bitrate = 0
    util.total_play_time = 0
    util.total_bitrate_change = 0
    util.total_log_bitrate_change = 0
    util.total_reaction_time = 0
    util.last_played = None

    overestimate_count = 0
    overestimate_average = 0
    goodestimate_count = 0
    goodestimate_average = 0
    estimate_average = 0

    util.rampup_origin = 0
    util.rampup_time = None
    util.rampup_threshold = rampup_threshold

    util.max_buffer_size = max_buffer * 1000

    util.manifest = util.load_json(movie)
    bitrates = util.manifest['bitrates_kbps']
    utility_offset = 0 - math.log(bitrates[0])  # so utilities[0] = 0
    utilities = [math.log(b) + utility_offset for b in bitrates]
    if movie_length != None:
        l1 = len(util.manifest['segment_sizes_bits'])
        l2 = math.ceil(movie_length * 1000 /
                       util.manifest['segment_duration_ms'])
        util.manifest['segment_sizes_bits'] *= math.ceil(l2 / l1)
        util.manifest['segment_sizes_bits'] = util.manifest['segment_sizes_bits'][0:l2]
    util.manifest = ManifestInfo(segment_time=util.manifest['segment_duration_ms'],
                                 bitrates=bitrates,
                                 utilities=utilities,
                                 segments=util.manifest['segment_sizes_bits'])
    SessionInfo.manifest = util.manifest

    network_trace = util.load_json(network)
    network_trace = [NetworkPeriod(time=p['duration_ms'],
                                   bandwidth=p['bandwidth_kbps'] *
                                   network_multiplier,
                                   latency=p['latency_ms'])
                     for p in network_trace]

    buffer_size = max_buffer * 1000
    gamma_p = gamma_p

    config = {'buffer_size': buffer_size,
              'gp': gamma_p,
              'abr_osc': abr_osc,
              'abr_basic': abr_basic,
              'no_ibr': no_insufficient_buffer_rule}
    if abr[-3:] == '.py':
        abr = AbrInput(abr, config)
    else:
        abr_list[abr].use_abr_o = abr_osc
        abr_list[abr].use_abr_u = not abr_osc
        abr = abr_list[abr](config, util)
    network = NetworkModel(network_trace, util)

    if replace[-3:] == '.py':
        replacer = ReplacementInput(replace)
    if replace == 'left':
        replacer = Replace(0)
    elif replace == 'right':
        replacer = Replace(1)
    else:
        replacer = NoReplace(util)

    config = {'window_size': window_size, 'half_life': half_life}
    throughput_history = average_list[moving_average](config, util)

    # download first segment
    quality = abr.get_first_quality()
    size = util.manifest.segments[0][quality]
    download_metric = network.download(size, 0, quality, 0)
    download_time = download_metric.time - download_metric.time_to_first_bit
    startup_time = download_time
    util.buffer_contents.append(download_metric.quality)
    t = download_metric.size / download_time
    l = download_metric.time_to_first_bit
    throughput_history.push(download_time, t, l)
    # print('%d,%d -> %d,%d' % (t, l, throughput, latency))
    util.total_play_time += download_metric.time

    if util.verbose:
        print('[%d-%d]  %d: q=%d s=%d/%d t=%d=%d+%d bl=0->0->%d' %
              (0, round(download_metric.time), 0, download_metric.quality,
               download_metric.downloaded, download_metric.size,
               download_metric.time, download_metric.time_to_first_bit,
               download_metric.time - download_metric.time_to_first_bit,
               util.get_buffer_level()))

     # download rest of segments
    next_segment = 1
    abandoned_to_quality = None
    while next_segment < len(util.manifest.segments):

        # TODO: BEGIN TODO: reimplement seeking - currently only proof-of-concept hack
        if seek != None:
            if next_segment * util.manifest.segment_time >= 1000 * seek[0]:
                next_segment = math.floor(
                    1000 * seek[1] / util.manifest.segment_time)
                util.buffer_contents = []
                util.buffer_fcc = 0
                abr.report_seek(1000 * seek[1])
                seek = None
                util.rampup_origin = util.total_play_time
                util.rampup_time = None
        # TODO:  END TODO:  reimplement seeking - currently only proof-of-concept hack

        # do we have space for a new segment on the buffer?
        full_delay = util.get_buffer_level() + util.manifest.segment_time - buffer_size
        if full_delay > 0:
            util.deplete_buffer(full_delay)
            network.delay(full_delay)
            abr.report_delay(full_delay)
            if util.verbose:
                print('full buffer delay %d bl=%d' %
                      (full_delay, util.get_buffer_level()))

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
        if no_abandon:
            check_abandon = None

        size = util.manifest.segments[current_segment][quality]

        if delay > 0:
            util.deplete_buffer(delay)
            network.delay(delay)
            if util.verbose:
                print('abr delay %d bl=%d' % (delay, util.get_buffer_level()))

        # print('size %d, current_segment %d, quality %d, buffer_level %d' %
        #      (size, current_segment, quality, get_buffer_level()))

        download_metric = network.download(size, current_segment, quality,
                                           util.get_buffer_level(), check_abandon)

        # print('index %d, quality %d, downloaded %d/%d, time %d=%d+.' %
        #      (download_metric.index, download_metric.quality,
        #       download_metric.downloaded, download_metric.size,
        #       download_metric.time, download_metric.time_to_first_bit))

        if util.verbose:
            print('[%d-%d]  %d: q=%d s=%d/%d t=%d=%d+%d ' %
                  (round(util.total_play_time), round(util.total_play_time + download_metric.time),
                   current_segment, download_metric.quality,
                   download_metric.downloaded, download_metric.size,
                   download_metric.time, download_metric.time_to_first_bit,
                   download_metric.time - download_metric.time_to_first_bit),
                  end='')
            if replace == None:
                if download_metric.abandon_to_quality == None:
                    print('bl=%d' % util.get_buffer_level(), end='')
                else:
                    print(' ABANDONED to %d - %d/%d bits in %d=%d+%d ttfb+ttdl  bl=%d' %
                          (download_metric.abandon_to_quality,
                           download_metric.downloaded, download_metric.size,
                           download_metric.time, download_metric.time_to_first_bit,
                           download_metric.time - download_metric.time_to_first_bit,
                           util.get_buffer_level()),
                          end='')
            else:
                if download_metric.abandon_to_quality == None:
                    print(' REPLACEMENT  bl=%d' %
                          util.get_buffer_level(), end='')
                else:
                    print(' REPLACMENT ABANDONED after %d=%d+%d ttfb+ttdl  bl=%d' %
                          (download_metric.time, download_metric.time_to_first_bit,
                           download_metric.time - download_metric.time_to_first_bit,
                           util.get_buffer_level()),
                          end='')

        # print('deplete buffer %d' % download_metric.time)
        util.deplete_buffer(download_metric.time)
        if util.verbose:
            print('->%d' % util.get_buffer_level(), end='')

        # update buffer with new download
        if replace == None:
            if download_metric.abandon_to_quality == None:
                util.buffer_contents += [quality]
                next_segment += 1
            else:
                abandon_to_quality = download_metric.abandon_to_quality
        else:
            # abandon_to_quality == None
            if download_metric.abandon_to_quality == None:
                if util.get_buffer_level() + util.manifest.segment_time * replace >= 0:
                    util.buffer_contents[replace] = quality
                else:
                    print('WARNING: too late to replace')
                    pass
            else:
                pass
            # else: do nothing because segment abandonment does not suggest new download

        if util.verbose:
            print('->%d' % util.get_buffer_level())

        abr.report_download(download_metric, replace != None)

        # calculate throughput and latency
        download_time = download_metric.time - download_metric.time_to_first_bit
        t = download_metric.downloaded / download_time
        l = download_metric.time_to_first_bit

        # check accuracy of throughput estimate
        if util.throughput > t:
            overestimate_count += 1
            overestimate_average += (util.throughput - t -
                                     overestimate_average) / overestimate_count
        else:
            goodestimate_count += 1
            goodestimate_average += (t - util.throughput -
                                     goodestimate_average) / goodestimate_count
        estimate_average += ((util.throughput - t - estimate_average) /
                             (overestimate_count + goodestimate_count))

        # update throughput estimate
        if download_metric.abandon_to_quality == None:
            throughput_history.push(download_time, t, l)

        # loop while next_segment < len(manifest.segments)

    util.playout_buffer()

    # multiply by to_time_average to get per/chunk average
    to_time_average = 1 / (util.total_play_time / util.manifest.segment_time)
    count = len(util.manifest.segments)
    time = count * util.manifest.segment_time + util.rebuffer_time + startup_time

    if verbose:
        print('buffer size: %d' % buffer_size)
        print('total played utility: %f' % util.played_utility)
        print('time average played utility: %f' %
            (util.played_utility * to_time_average))
        print('total played bitrate: %f' % util.played_bitrate)
        print('time average played bitrate: %f' %
            (util.played_bitrate * to_time_average))
        print('total play time: %f' % (util.total_play_time / 1000))
        print('total play time chunks: %f' %
            (util.total_play_time / util.manifest.segment_time))
        print('total rebuffer: %f' % (util.rebuffer_time / 1000))
        print('rebuffer ratio: %f' % (util.rebuffer_time / util.total_play_time))
        print('time average rebuffer: %f' %
            (util.rebuffer_time / 1000 * to_time_average))
        print('total rebuffer events: %f' % util.rebuffer_event_count)
        print('time average rebuffer events: %f' %
            (util.rebuffer_event_count * to_time_average))
        print('total bitrate change: %f' % util.total_bitrate_change)
        print('time average bitrate change: %f' %
            (util.total_bitrate_change * to_time_average))
        print('total log bitrate change: %f' % util.total_log_bitrate_change)
        print('time average log bitrate change: %f' %
            (util.total_log_bitrate_change * to_time_average))
        print('time average score: %f' %
            (to_time_average * (util.played_utility -
                                gamma_p * util.rebuffer_time / util.manifest.segment_time)))

        if overestimate_count == 0:
            print('over estimate count: 0')
            print('over estimate: 0')
        else:
            print('over estimate count: %d' % overestimate_count)
            print('over estimate: %f' % overestimate_average)
        if goodestimate_count == 0:
            print('leq estimate count: 0')
            print('leq estimate: 0')
        else:
            print('leq estimate count: %d' % goodestimate_count)
            print('leq estimate: %f' % goodestimate_average)
        print('estimate: %f' % estimate_average)
        if util.rampup_time == None:
            print('rampup time: %f' %
                (len(util.manifest.segments) * util.manifest.segment_time / 1000))
        else:
            print('rampup time: %f' % (util.rampup_time / 1000))
        print('total reaction time: %f' % (util.total_reaction_time / 1000))

    results_dict = {
        'buffer_size': buffer_size,
        'total_played_utility': util.played_utility,
        'time_average_played_utility': util.played_utility * to_time_average,
        'total_played_bitrate': util.played_bitrate,
        'time_average_played_bitrate': util.played_bitrate * to_time_average,
        'total_play_time': util.total_play_time / 1000,
        'total_play_time_chunks': util.total_play_time / util.manifest.segment_time,
        'total_rebuffer': util.rebuffer_time / 1000,
        'rebuffer_ratio': util.rebuffer_time / util.total_play_time,
        'time_average_rebuffer': util.rebuffer_time / 1000 * to_time_average,
        'total_rebuffer_events': util.rebuffer_event_count,
        'time_average_rebuffer_events': util.rebuffer_event_count * to_time_average,
        'total_bitrate_change': util.total_bitrate_change,
        'time_average_bitrate_change': util.total_bitrate_change * to_time_average,
        'total_log_bitrate_change': util.total_log_bitrate_change,
        'time_average_log_bitrate_change': util.total_log_bitrate_change * to_time_average,
        'time_average_score': to_time_average * (util.played_utility - gamma_p * util.rebuffer_time / util.manifest.segment_time),
        'total_reaction_time': util.total_reaction_time / 1000,
        'estimate': estimate_average,
    }
    if verbose:
        print(results_dict)

    return results_dict


if __name__ == '__main__':
    init()
