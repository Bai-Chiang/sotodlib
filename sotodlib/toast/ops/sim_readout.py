# Copyright (c) 2018-2024 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

import os
import pickle

import ephem
import h5py
import healpy as hp
import numpy as np
import traitlets
from astropy import constants
from astropy import units as u
from scipy.constants import au as AU
from scipy.interpolate import RectBivariateSpline
from toast import qarray as qa
from toast.data import Data
from toast.observation import default_values as defaults
from toast.ops.operator import Operator
from toast.timing import function_timer, Timer
from toast.traits import (Bool, Float, Instance, Int, Quantity, Unicode,
                          trait_docs)
from toast.utils import Environment, Logger, unit_conversion, name_UID

XAXIS, YAXIS, ZAXIS = np.eye(3)


@trait_docs
class SimReadout(Operator):
    """Simulate various readout-related systematics"""

    API = Int(0, help="Internal interface version for this operator")

    realization = Int(0, help="The noise realization index")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask_readout = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional shared flagging"
    )

    det_data = Unicode(
        defaults.det_data,
        help="Observation detdata key for simulated signal",
    )

    glitch_rate = Quantity(0.1 * u.Hz, help="Glitch occurrence rate")

    glitch_amplitude_center = Quantity(
        1 * u.K, help="Center of Gaussian distribution for glitch amplitude"
    )

    glitch_amplitude_sigma = Quantity(
        1 * u.K, help="Width of Gaussian distribution for glitch amplitude"
    )

    bias_line_glitch_rate = Quantity(0.1 * u.Hz, help="Glitch occurrence rate")

    bias_line_glitch_amplitude_center = Quantity(
        1 * u.K, help="Center of Gaussian distribution for glitch amplitude"
    )

    bias_line_glitch_amplitude_sigma = Quantity(
        1 * u.K, help="Width of Gaussian distribution for glitch amplitude"
    )

    bias_line_glitch_amplitude_scatter = Float(
        1.0,
        help="Width of Gaussian distribution for *relative* glitch amplitude.  "
        "Negative factors get cropped."
    )

    cosmic_ray_glitch_rate = Quantity(
        0.000001 * u.Hz / u.mm**2,
        help="Cosmic ray glitch rate *either* per unit focalplane area "
        "e.g. [Hz / mm^2] *or* rate per the entire focalplane [Hz]",
    )

    cosmic_ray_glitch_amplitude_center = Quantity(
        1 * u.K, help="Center of Gaussian distribution for glitch amplitude"
    )

    cosmic_ray_glitch_amplitude_sigma = Quantity(
        1 * u.K, help="Width of Gaussian distribution for glitch amplitude"
    )

    cosmic_ray_correlation_length = Quantity(
        10 * u.mm, help="Distance scale of cosmic ray events on the focalplane"
    )

    cosmic_ray_time_constant = Quantity(
        1.0 * u.s, help="Dissipation time of cosmic ray events on the focaplane"
    )

    simulate_glitches = Bool(True, help="Enable glitch simulation")

    jump_rate = Quantity(0.001 * u.Hz, help="Jump occurrence rate")

    jump_amplitude_center = Quantity(
        0 * u.mK, help="Center of Gaussian distribution for jump amplitude"
    )

    jump_amplitude_sigma = Quantity(
        1 * u.mK, help="Width of Gaussian distribution for jump amplitude"
    )

    bias_line_jump_rate = Quantity(0.001 * u.Hz, help="Jump occurrence rate")

    bias_line_jump_amplitude_center = Quantity(
        0 * u.mK, help="Center of Gaussian distribution for jump amplitude"
    )

    bias_line_jump_amplitude_sigma = Quantity(
        1 * u.mK, help="Width of Gaussian distribution for jump amplitude"
    )

    bias_line_jump_amplitude_scatter = Float(
        1.0,
        help="Width of Gaussian distribution for *relative* jump amplitude.  "
        "Negative factors get cropped."
    )

    simulate_jumps = Bool(True, help="Enable jump simulation")

    misidentify_bolometers = Bool(True, help="Enable bolometer misidentification")

    misidentification_width = Quantity(
        1.0 * u.deg,
        help="Probability of misidentifying two channels is a Gaussian "
        "function of their distance on the focalplane.",
    )

    yield_center = Float(
        0.8, help="Center of Gaussian distribution for detector yield"
    )

    yield_sigma = Float(
        0.1, help="Width of Gaussian distribution for detector yield"
    )

    simulate_yield = Bool(True, help="Simulate failed bolometers")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _simulate_events(self, seed, times, rate):
        """Simulate a Poissonian Process based on rate.
        times: np.array of time stamps
        rate: event occurrence rate in unit Hz
        """
        np.random.seed(int(seed) % 2**32)
        event_time = times[0]
        event_indices = []
        while event_time < times[-1]:
            # Sample next event based on the interval between two events
            # which follows exponential distribution.
            interval = np.random.exponential(scale=1.0/rate)
            event_time += interval
            if event_time < times[-1]:
                # Find the time stamp cloest to event_time
                event_indices.append(
                        np.argmin(np.abs( times-event_time ))
                        )
        event_counts = len(event_indices)
        return event_counts, event_indices

    def _simulate_amplitudes(
            self, seed, event_counts, amplitude_center, amplitude_sigma
    ):
        """Simulate Gaussian event amplitudes"""
        # Drawing amplitudes for every affected sample is not the
        # same as drawing amplitudes for every event but it is fast.
        # If we routinely have more than one event occurring during
        # the same time stamp we are in trouble
        np.random.seed(int(seed) % 2**32)
        amplitudes = amplitude_center + np.random.randn(event_counts) * amplitude_sigma
        return amplitudes

    def _add_uncorrelated_glitches(
            self, comm, signal, obs_id, focalplane, fsample, times, local_dets
    ):
        """Simulate uncorrelated electronic glitches"""
        log = Logger.get()

        ndets = len(local_dets)
        nglitch = 0
        for det in local_dets:
            det_id = focalplane[det]["uid"]
            sig = signal[det]
            event_counts, event_indices = self._simulate_events(
                obs_id + det_id  + self.realization + 8123765,
                times,
                self.glitch_rate.to_value(u.Hz),
            )
            events = self._simulate_amplitudes(
                obs_id + det_id  + self.realization + 150243972,
                event_counts,
                self.glitch_amplitude_center.to_value(signal.units),
                self.glitch_amplitude_sigma.to_value(signal.units),
            )
            # Add delta-function glitches
            sig[event_indices] += events
            nglitch += event_counts

        if comm is not None:
            nglitch = comm.allreduce(nglitch)
            ndets = comm.allreduce(ndets)
        msg = "Simulated {} uncorrelated glitches with average rate {:.5f} Hz".format(
                nglitch,
                nglitch/((times[-1]-times[0])*ndets)
                )
        log.debug_rank(msg, comm=comm)

        return

    def _add_correlated_glitches(
            self, comm, signal, obs_id, focalplane, fsample, times, bias2det
    ):
        """Simulate bias line glitches"""
        log = Logger.get()

        nbias = 0
        nglitch = 0
        for tube, wafers in bias2det.items():
            tube_id = name_UID(tube)
            for wafer, biases in wafers.items():
                wafer_id = name_UID(wafer)
                nbias += len(biases)
                for bias, dets in biases.items():
                    # The event locations on the bias line are shared
                    event_counts, event_indices = self._simulate_events(
                        obs_id + tube_id + wafer_id + bias  + self.realization \
                        + 5295629,
                        times,
                        self.bias_line_glitch_rate.to_value(u.Hz),
                    )
                    # Base amplitude is shared ...
                    events = self._simulate_amplitudes(
                        obs_id + tube_id + wafer_id  + self.realization + 150243972,
                        event_counts,
                        self.bias_line_glitch_amplitude_center.to_value(signal.units),
                        self.bias_line_glitch_amplitude_sigma.to_value(signal.units),
                    )
                    # ... but modulated at the detector level
                    for det in dets:
                        det_id = focalplane[det]["uid"]
                        sig = signal[det]
                        scale = self._simulate_amplitudes(
                            obs_id + det_id  + self.realization + 74712734,
                            event_counts,
                            1,
                            self.bias_line_glitch_amplitude_scatter,
                        )
                        scale[scale < 0] = 0
                        # Add delta-function glitches
                        sig[event_indices] += events * scale
                    nglitch += event_counts

        if comm is not None:
            nglitch = comm.allreduce(nglitch)
            nbias = comm.allreduce(nbias)
        msg = "Simulated {} correlated glitches with average rate {:.5f} Hz".format(
                nglitch,
                nglitch/((times[-1]-times[0])*nbias)
                )
        log.debug_rank(msg, comm=comm)

        return

    def _add_cosmic_ray_glitches(
            self,
            comm,
            signal,
            obs_id,
            focalplane,
            fsample,
            times,
            local_dets,
    ):
        """Simulate cosmic ray glitches

        This is the only glitch population to be convolved with a time constant
        """
        log = Logger.get()

        platescale = focalplane.detector_data.meta["platescale"].to_value(u.deg / u.mm)
        fov = focalplane.field_of_view.to_value(u.deg)
        diameter = fov / platescale

        # Map detectors to their physical position on the focalplane
        det2position = {}
        xrot = qa.rotation(YAXIS, np.pi / 2)
        for det in local_dets:
            quat = focalplane[det]["quat"]
            vec = qa.rotate(qa.mult(xrot, quat), ZAXIS)
            lon, lat = hp.vec2dir(vec, lonlat=True)
            x = lon / platescale
            y = lat / platescale
            det2position[det] = (x, y)


        # Glitches are simulated over a square grid and individual
        # detectors are affected based on their proximity to the event
        area = diameter**2
        ratio = np.pi / 4  # square to disc area ratio
        try:
            # Did the user supply a rate for the entire focaplane?
            glitch_rate = self.cosmic_ray_glitch_rate.to_value(u.Hz)
            # Increase the rate to sample the glitches across a square area
            glitch_rate /= ratio
        except u.UnitConversionError:
            # Nope, it must be per unit area
            glitch_rate = self.cosmic_ray_glitch_rate.to_value(u.Hz / u.mm**2)
            glitch_rate *= area

        log.debug_rank(
            f"FOV = {fov:.2f} deg, platescale = {platescale:.6f} deg/mm, \n"
            f"focalplane diameter = {diameter} mm, area = {area * ratio:.3f} mm^2\n"
            f"glitch_rate = {glitch_rate / area:.6f} Hz/mm^2, \n"
            f"glitch_rate = {glitch_rate * ratio:.6f} Hz / focalplane",
            comm=comm,
        )

        event_counts, event_indices = self._simulate_events(
            obs_id + self.realization + 98552896,
            times,
            glitch_rate,
        )

        # Generate a glitch profile that we will re-use
        tau = self.cosmic_ray_time_constant
        wtemplate = int(10 * tau.to_value(u.s) * fsample.to_value(u.Hz))
        log.debug_rank(f"Width of the glitch template is {wtemplate}", comm=comm)
        t = np.arange(wtemplate) / fsample.to_value(u.Hz)
        template = np.exp(-t / tau.to_value(u.s))

        # For every event, we need a location
        sqradius = (0.5 * diameter)**2
        center = self.cosmic_ray_glitch_amplitude_center.to_value(signal.units)
        sigma = self.cosmic_ray_glitch_amplitude_sigma.to_value(signal.units)
        corr_length = self.cosmic_ray_correlation_length.to_value(u.mm)
        nglitch = 0
        ndet = 0
        for start, count in zip(event_indices, event_counts):
            stop = min(start + wtemplate, nsample)
            ind = slice(start, stop)
            templ = template[:stop - start]
            for i in range(count):
                amp = center + np.random.randn() * sigma
                x = (np.random.rand() - .5) * diameter
                y = (np.random.rand() - .5) * diameter
                if x**2 + y**2 > sqradius:
                    continue
                nglitch += 1
                # Observe the glitch with the local detectors
                for det in local_dets:
                    xdet, ydet = det2position[det]
                    dist = np.sqrt((x - xdet)**2 + (y - ydet)**2)
                    if dist > 4 * corr_length:
                        # This detector is to far to register
                        continue
                    ndet += 1
                    amp_det = amp * np.exp(-dist / corr_length)
                    signal[det][ind] += amp_det * templ

        if comm is not None:
            ndet = comm.allreduce(ndet)
        log.debug_rank(
            f"Simulated {nglitch} cosmic ray glitches and {ndet} "
            f"detector events",
            comm=comm,
        )

        return

    @function_timer
    def _add_glitches(
            self,
            comm,
            obs_id,
            times,
            focalplane,
            signal,
            all_dets,
            local_dets,
            bias2det,
    ):
        """Simulate glitches"""
        if not self.simulate_glitches:
            return

        fsample = focalplane.sample_rate

        if self.glitch_rate != 0:
            self._add_uncorrelated_glitches(
                comm, signal, obs_id, focalplane, fsample, times, local_dets
            )

        if self.bias_line_glitch_rate != 0:
            self._add_correlated_glitches(
                comm, signal, obs_id, focalplane, fsample, times, bias2det
            )

        if self.cosmic_ray_glitch_rate != 0:
            self._add_cosmic_ray_glitches(
                comm,
                signal,
                obs_id,
                focalplane,
                fsample,
                times,
                local_dets,
            )

        return

    @function_timer
    def _add_jumps(
            self,
            comm,
            obs_id,
            times,
            focalplane,
            signal,
            all_dets,
            local_dets,
            bias2det,
    ):
        """Simulate baseline jumps"""
        if not self.simulate_jumps:
            return
        log = Logger.get()
        fsample = focalplane.sample_rate

        # Uncorrelated jumps
        njump = 0
        for det in local_dets:
            det_id = focalplane[det]["uid"]
            sig = signal[det]
            event_counts, event_indices = self._simulate_events(
                obs_id + det_id  + self.realization + 45835364,
                times,
                self.jump_rate.to_value(u.Hz),
            )
            events = self._simulate_amplitudes(
                obs_id + det_id  + self.realization + 250243972,
                event_counts,
                self.jump_amplitude_center.to_value(signal.units),
                self.jump_amplitude_sigma.to_value(signal.units),
            )
            # Change baseline at every event
            for index, event in zip(event_indices, events):
                sig[index:] += event
            njump += event_counts

        if comm is not None:
            njump = comm.allreduce(njump)
        log.debug_rank(f"Simulated {njump} uncorrelated jumps", comm=comm)

        # Bias line jumps
        njump = 0
        for tube, wafers in bias2det.items():
            tube_id = name_UID(tube)
            for wafer, biases in wafers.items():
                wafer_id = name_UID(wafer)
                for bias, dets in biases.items():
                    # The event locations on the bias line are shared
                    event_counts, event_indices = self._simulate_events(
                        obs_id + tube_id + wafer_id + bias  + self.realization + 936582,
                        times,
                        self.bias_line_jump_rate.to_value(u.Hz),
                    )
                    # Base amplitude is shared ...
                    events = self._simulate_amplitudes(
                        obs_id + tube_id + wafer_id  + self.realization + 1295723,
                        event_counts,
                        self.bias_line_jump_amplitude_center.to_value(signal.units),
                        self.bias_line_jump_amplitude_sigma.to_value(signal.units),
                    )
                    # ... but modulated at the detector level
                    for det in dets:
                        det_id = focalplane[det]["uid"]
                        sig = signal[det]
                        scale = self._simulate_amplitudes(
                            obs_id + det_id  + self.realization + 83276345,
                            event_counts,
                            1,
                            self.bias_line_jump_amplitude_scatter,
                        )
                        scale[scale < 0] = 0
                        # Change baseline at every event
                        for index, event in zip(event_indices, events * scale):
                            sig[index:] += event
                    njump += event_counts

        if comm is not None:
            njump = comm.allreduce(njump)
        log.debug_rank(f"Simulated {njump} correlated jumps", comm=comm)

        return

    @function_timer
    def _fail_bolometers(
            self,
            comm,
            obs_id,
            times,
            focalplane,
            signal,
            all_dets,
            local_dets,
            bias2det,
    ):
        """Simulate bolometers that fail to bias or otherwise are unusable"""
        if not self.simulate_yield:
            return
        log = Logger.get()
        np.random.seed(int(obs_id + self.realization + 59127574) % 2**32)
        yield_ = self.yield_center + np.random.randn() * self.yield_sigma
        yield_ = min(1, max(0, yield_))
        nfail = 0
        for det in local_dets:
            det_id = focalplane[det]["uid"]
            np.random.seed(int(obs_id + self.realization + 59127574 + det_id) % 2**32)
            x = np.random.rand()
            if x > yield_:
                # FIXME: We should choose between various failure modes here
                #
                # Fail the bolometer by replacing the signal with
                # a linear trend.  We do *not* record the failure in the
                # detector flags so the data *cannot* be reduced without
                # some sort of mitigation
                start, stop = np.random.rand(2) * 10 * u.K
                x = (times - times[0]) / (times[-1] - times[0])
                trend = start + (stop - start) * x
                signal[det] = trend
                nfail += 1

        if comm is not None:
            nfail = comm.allreduce(nfail)
        ndet = len(all_dets)
        log.debug_rank(
            f"Failed {nfail} / {ndet} = {nfail/ndet:.3f} bolometers.  "
            f"Target yield was {yield_:.3f}",
            comm=comm,
        )
        return

    @function_timer
    def _misidentify_bolometers(
            self,
            comm,
            obs_id,
            times,
            focalplane,
            signal,
            all_dets,
            local_dets,
            det2rank,
    ):
        """Swap detector data between two confused readout channels"""
        if not self.misidentify_bolometers:
            return
        log = Logger.get()
        np.random.seed(int(obs_id + self.realization) % 2**32)
        nhit = 0
        for det1 in all_dets:
            for det2 in all_dets:
                if det1 == det2:
                    continue
                matched = True
                # Misidentification requires several properties to agree
                for prop in ["band", "card_slot", "wafer_slot", "bias", "pol"]:
                    if focalplane[det1][prop] != focalplane[det2][prop]:
                        matched = False
                        break
                if not matched:
                    continue
                # Detectors are matched, see if they get misidentified
                quat1 = focalplane[det1]["quat"]
                quat2 = focalplane[det2]["quat"]
                vec1 = qa.rotate(quat1, ZAXIS)
                vec2 = qa.rotate(quat2, ZAXIS)
                dist = np.arccos(np.dot(vec1, vec2))
                x = np.random.randn() * self.misidentification_width.to_value(u.rad)
                if np.abs(x) > dist:
                    nhit += 1
                    # Swap the signals.  This may require MPI communication
                    if det1 in local_dets and det2 in local_dets:
                        # Local operation
                        temp = signal[det1].copy()
                        signal[det1] = signal[det2]
                        signal[det2] = temp
                    elif det1 in local_dets:
                        # Send and receive signal
                        target = det2rank[det2]
                        comm.Send(signal[det1], dest=target, tag=nhit)
                        comm.Recv(signal[det1], source=target, tag=nhit)
                    elif det2 in local_dets:
                        # Receive and send signal
                        target = det2rank[det1]
                        sig2 = signal[det2].copy()
                        comm.Recv(signal[det2], source=target, tag=nhit)
                        comm.Send(sig2, dest=target, tag=nhit)
        ndet = len(all_dets)
        npair = ndet * (ndet - 1) // 2
        log.debug_rank(f"Misidentified {nhit} / {npair} detector pairs", comm=comm)
        return

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        for trait in "shared_flags", "det_data":
            value = getattr(self, trait)
            if value is None:
                raise RuntimeError(f"You must set `{trait}` before running SimReadout")

        for obs in data.obs:
            comm = obs.comm.comm_group
            obs_id = obs.uid
            times = obs.shared[self.times].data
            all_dets = obs.all_detectors
            local_dets = obs.select_local_detectors(detectors)
            obs.detdata.ensure(self.det_data, detectors=local_dets, create_units=u.K)

            # Map detectors to their owning processes
            det2rank = {}
            for rank, dets in enumerate(obs.dist.dets):
                for det in dets:
                    det2rank[det] = rank
            focalplane = obs.telescope.focalplane
            signal = obs.detdata[self.det_data]

            if 'tube_slot' in focalplane.detector_data.keys():
                tube_slot_key = 'tube_slot'
            elif 'det_info:tel_tube' in focalplane.detector_data.keys():
                tube_slot_key = 'det_info:tel_tube'
            else:
                raise RuntimeError("Can't find 'tude_slot' or 'det_info:tel_tube' in focalplane.detector_data")

            if 'wafer_slot' in focalplane.detector_data.keys():
                wafer_slot_key = 'wafer_slot'
            elif 'det_info:wafer_slot' in focalplane.detector_data.keys():
                wafer_slot_key = 'det_info:wafer_slot'
            else:
                raise RuntimeError("Can't find 'wafer_slot' or 'det_info:wafer_slot' in focalplane.detector_data")

            if 'bias' in focalplane.detector_data.keys():
                bias_key = 'bias'
            elif 'det_info:wafer:bias_line' in focalplane.detector_data.keys():
                bias_key = 'det_info:wafer:bias_line'
            else:
                raise RuntimeError("Can't find 'bias' or 'det_info:wafer:bias_line' in focalplane.detector_data")

            # Map bias lines to detectors
            bias2det = {}
            for det in local_dets:
                tube = focalplane[det][tube_slot_key]
                if tube not in bias2det:
                    bias2det[tube] = {}
                wafer = focalplane[det][wafer_slot_key]
                if wafer not in bias2det[tube]:
                    bias2det[tube][wafer] = {}
                bias = focalplane[det][bias_key]
                if bias not in bias2det[tube][wafer]:
                    bias2det[tube][wafer][bias] = []
                bias2det[tube][wafer][bias].append(det)

            self._add_glitches(
                comm,
                obs_id,
                times,
                focalplane,
                signal,
                all_dets,
                local_dets,
                bias2det,
            )
            self._add_jumps(
                comm, obs_id, times, focalplane, signal, all_dets, local_dets, bias2det
            )
            self._fail_bolometers(
                comm, obs_id, times, focalplane, signal, all_dets, local_dets, bias2det
            )
            self._misidentify_bolometers(
                comm, obs_id, times, focalplane, signal, all_dets, local_dets, det2rank
            )

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "shared": [self.times, self.shared_flags],
            "detdata": [self.det_data],
        }
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": [],
            "detdata": [self.det_data],
        }
        return prov

    def _accelerators(self):
        return list()
