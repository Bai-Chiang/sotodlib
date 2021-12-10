#!/usr/bin/env python3

import so3g
from spt3g import core
import numpy as np

def pos2vel(p):
    return np.ediff1d(p)

def locate_sign_changes(t, dy=0.001, min_gap=200):
    tmin = np.min(t)
    tmax = np.max(t)

    if len(t) == 0:
        return []

    if np.sign(tmax) == np.sign(tmin):
        return []

    # If the data does not entirely cross the threshold region,
    # do not consider it a sign change
    if tmin > -dy and tmax < dy:
        return []

    # Find where the data crosses the lower and upper boundaries of the threshold region
    c_lower = (np.where(np.sign(t[:-1]+dy) != np.sign(t[1:]+dy))[0] + 1)
    c_upper = (np.where(np.sign(t[:-1]-dy) != np.sign(t[1:]-dy))[0] + 1)

    # Noise handling:
    # If there are multiple crossings of the same boundary (upper or lower) in quick
    # succession (i.e., less than min_gap), it is mostly likely due to noise. In this case,
    # take the average of each group of crossings.
    if len(c_lower) > 0:
        spl = np.array_split(c_lower, np.where(np.ediff1d(c_lower) > min_gap)[0] + 1)
        c_lower = np.array([int(np.ceil(np.mean(s))) for s in spl])
    if len(c_upper) > 0:
        spu = np.array_split(c_upper, np.where(np.ediff1d(c_upper) > min_gap)[0] + 1)
        c_upper = np.array([int(np.ceil(np.mean(s))) for s in spu])

    events = np.sort(np.concatenate((c_lower, c_upper)))

    # Look for zero-crossings
    zc = []
    while len(c_lower) > 0 and len(c_upper) > 0:
        # Crossing from -ve to +ve
        if c_lower[0] < c_upper[0]:
            b = c_lower[c_lower < c_upper[0]]
            zc.append(int( np.ceil(np.mean([b[-1], c_upper[0]])) ))
            c_lower = c_lower[len(b):]
        # Crossing from +ve to -ve
        elif c_upper[0] < c_lower[0]:
            b = c_upper[c_upper < c_lower[0]]
            zc.append(int( np.ceil(np.mean([b[-1], c_lower[0]])) ))
            c_upper = c_upper[len(b):]

    # Replace all upper and lower crossings that contain a zero-crossing in between with the
    # zero-crossing itself, but ONLY if those three events happen in quick succession (i.e.,
    # shorter than min_gap). Otherwise, they are separate events -- there is likely a stop
    # state in between; in this case, do NOT perform the replacement.
    for z in zc:
        before_z = events[events < z]
        after_z  = events[events > z]
        if (after_z[0] - before_z[-1]) < min_gap:
            events = np.concatenate((before_z[:-1], [z], after_z[1:]))

    # If the last event is close to the end, a crossing of the upper or lower threshold
    # boundaries may or may not be a significant event -- there is not enough remaining data
    # to determine whether it is stopping or not. This will be clarified in the next iteration
    # of the loop, when more HK data will have been added. (Or if there is no more HK data to
    # follow, the loss in information by removing the last crossing is negligible.)
    # On the other hand, if the last event is a zero-crossing, then that is unambiguous and
    # should not be removed.
    if (len(t) - events[-1] < min_gap) and events[-1] not in zc:
        events = events[:-1]

    # A similar problem occurs when the first boundary-crossing event is close to the
    # beginning -- it could either be following a zero-crossing OR be exiting a stopped state.
    # In this case, we use previous data to disambiguate the two cases, and is deferred to
    # another function.

    return events

class _DataBundle():
    def __init__(self):
        self.times = []
        self.data = None

    def add(self, b):
        self.times.extend(b.times)
        if self.data is None:
            self.data = {c: [] for c in b.keys()}
        for c in b.keys():
            self.data[c].extend(b[c])

    def rebundle(self, flush_time):
        if len(self.times) == 0:
            return None

        output = core.G3TimesampleMap()
        output.times = core.G3VectorTime([t for t in self.times if t < flush_time])
        self.times = [t for t in self.times if t >= flush_time]

        for c in self.data.keys():
            output[c] = core.G3Timestream(np.array(self.data[c][:len(output.times)]))

        self.data = {c: self.data[c][len(output.times):] for c in self.data.keys()}

        return output

class _HKBundle(_DataBundle):
    def __init__(self):
        super().__init__()
        self.turnaround_times = []

    def set_azimuth_velocity(self):
        self.data['Azimuth_Velocity'] = pos2vel(self.data['Azimuth_Corrected'])

    def set_turnaround_times(self):
        if 'Azimuth_Velocity' not in self.data.keys():
            self.set_azimuth_velocity()
        self.turnaround_times = [self.times[i] for i in
                                 locate_sign_changes(self.data['Azimuth_Velocity'])]

    def ready(self):
        return len(self.turnaround_times) > 0

class _SmurfBundle(_DataBundle):
    def ready(self, flush_time):
        """
        Returns True if the current frame has crossed the flush_time
        """
        return len(self.times) > 0 and self.times[-1] >= flush_time

class FrameProcessor(object):
    def __init__(self):
        self.hkbundle = None
        self.smbundle = None
        self.flush_time = None
        self.maxlength = 10000
        self.current_state = 0  # default to scan state

    def ready(self):
        """
        Check if criterion passed in HK (for now, sign change in Az scan velocity data)
        """
        return self.hkbundle.ready() if (self.hkbundle is not None) else False

    def determine_state(self, v, dy=0.001):
        # If the velocity lies entirely in the threshold region, the telescope is stopped
        self.current_state = int(np.min(v) > -dy and np.max(v) < dy)

    def split_frame(self, f, maxlength=10000):
        output = []

        smb = _SmurfBundle()
        smb.add(f['data'])

        hkb = _HKBundle()
        hkb.add(f['hk'])

        while len(smb.times) > maxlength:
            t = smb.times[maxlength]

            g = core.G3Frame(core.G3FrameType.Scan)
            g['data'] = smb.rebundle(t)
            g['hk'] = hkb.rebundle(t)
            g['state'] = self.current_state

            output += [g]

        g = core.G3Frame(core.G3FrameType.Scan)
        g['data'] = smb.rebundle(smb.times[-1] + 1)
        g['hk'] = hkb.rebundle(hkb.times[-1] + 1)
        g['state'] = self.current_state

        output += [g]

        return output

    def flush(self):
        output = []

        f = core.G3Frame(core.G3FrameType.Scan)
        f['data'] = self.smbundle.rebundle(self.flush_time)
        f['hk'] = self.hkbundle.rebundle(self.flush_time)

        # Co-sampled (interpolated) azimuth encoder data
        f['data']['Azimuth'] = core.G3Timestream(np.interp(f['data'].times, f['hk'].times, f['hk']['Azimuth_Corrected'], left=np.nan, right=np.nan))

        self.determine_state(f['hk']['Azimuth_Velocity'])
        f['state'] = self.current_state

        if len(f['data'].times) > self.maxlength:
            output += self.split_frame(f, maxlength=self.maxlength)
        else:
            output += [f]

        return output

    def __call__(self, f):
        """
        Process a frame
        """
        if f.type != core.G3FrameType.Housekeeping and f.type != core.G3FrameType.Scan:
            return [f]

        if f.type == core.G3FrameType.Housekeeping:
            if self.hkbundle is None:
                self.hkbundle = _HKBundle()

            self.hkbundle.add(f['blocks'][0])   # 0th block for now
            self.hkbundle.set_azimuth_velocity()
            self.hkbundle.set_turnaround_times()

            # If the first detected turnaround (threshold crossing) event occurs near the
            # beginning of the frame, it is usually because it directly follows a sign-change
            # event and therefore NOT a significant event and should be ignored.
            # The exception is when the telescope is stopped: a threshold crossing after a stop
            # state IS a significant event, since it means the telescope is no longer stopped
            # -- in this case do NOT ignore it.
            if len(self.hkbundle.turnaround_times) > 0 and len(self.hkbundle.times) > 0:
                # G3Time is in units of 10 nanoseconds; 1 second = 1e8 units of time
                if int(self.hkbundle.turnaround_times[0]) - int(self.hkbundle.times[0]) < 1e8:
                    # This check occurs BEFORE the new state is determined, so current_state
                    # refers to the state at the end of the previous frame
                    if self.current_state == 0:
                        self.hkbundle.turnaround_times.pop(0)

        if f.type == core.G3FrameType.Scan:
            if self.smbundle is None:
                self.smbundle = _SmurfBundle()

            output = []

            self.smbundle.add(f['data'])

            if self.smbundle.ready(self.flush_time):
                output += self.flush()

            return output

class Bookbinder(object):
    """
    Bookbinder
    """
    def __init__(self, smurf_files, out_files):
        self._smurf_files = smurf_files
        self._out_files = out_files

        self.frameproc = FrameProcessor()
        self.smurf_iter = core.G3File(self._smurf_files.pop(0))
        self.writer = core.G3Writer(self._out_files.pop(0))

    def write_frames(self, frames_list):
        """
        Write frames to file
        """
        if not isinstance(frames_list, list):
            frames_list = list(frames_list)

        for f in frames_list:
            self.writer.Process(f)

    def __call__(self, fr):
        """
        Main loop

        Strategy:
        1. Add HK frames until sign change detected - get list of (timestamps of) sign changes
        2. While len(sign_changes) > 0, pop the first sign change, add SMuRF frames until that time;
        emit new frame; truncate HK and SMuRF frames
        3. Repeat Step 2 until len(sign_changes) == 0, then go back to Step 1
        """
        if fr.type != core.G3FrameType.Housekeeping:
            return

        if fr['hkagg_type'] != 2:
            return

        self.frameproc(fr)
        if not self.frameproc.ready():
            return

        tt = self.frameproc.hkbundle.turnaround_times
        output = []
        while len(tt) > 0:
            self.frameproc.flush_time = tt.pop(0)

            if self.frameproc.smbundle is not None and self.frameproc.smbundle.ready(self.frameproc.flush_time):
                output += self.frameproc.flush()

            while self.frameproc.smbundle is None or not self.frameproc.smbundle.ready(self.frameproc.flush_time):
                try:
                    f = next(self.smurf_iter)
                except StopIteration:
                    # If there are no more SMuRF frames, output remaining SMuRF data
                    if len(self.frameproc.smbundle.times) > 0:
                        self.frameproc.flush_time = self.frameproc.smbundle.times[-1] + 1  # +1 to ensure last sample gets included (= 1e-8 sec << sampling cadence)
                        output += self.frameproc.flush()
                    self.write_frames(output)

                    # If there are remaining files, update the
                    # SMuRF source iterator and G3 file writer
                    if len(self._smurf_files) > 0:
                        self.smurf_iter = core.G3File(self._smurf_files.pop(0))
                    if len(self._out_files) > 0:
                        self.writer = core.G3Writer(self._out_files.pop(0))

                    return

                if f.type != core.G3FrameType.Scan:
                    continue
                output += self.frameproc(f)  # FrameProcessor returns a list of frames (can be empty)

        self.write_frames(output)
        return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--g3', dest='g3file', nargs='+', type=str, required=True,
                        help='full path to G3 file')
    parser.add_argument('--hk', dest='hkfile', nargs='+', type=str, required=True,
                        help='full path to HK file')
    args = parser.parse_args()

    B = Bookbinder(args.g3file, ['out{:03d}.g3'.format(i) for i in range(len(args.g3file))])
    for h in core.G3File(args.hkfile):
        B(h)
