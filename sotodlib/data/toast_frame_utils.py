# Copyright (c) 2018-2019 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.
"""TOAST frame conversion utilities.
"""
import sys
import re

import itertools
import operator

import numpy as np

# Import so3g first so that it can control the import and monkey-patching
# of spt3g.  Then our import of spt3g_core will use whatever has been imported
# by so3g.
import so3g
from spt3g import core as core3g

from toast.mpi import MPI

import toast.qarray as qa
from toast.tod import spt3g_utils as s3utils


def frames_to_tod(
        tod,
        frame,
        frame_offset,
        frame_size,
        frame_data=None,
        detector_map="signal",
        flag_map="flags"):
    """Distribute a frame from the rank zero process.

    Args:
        tod (toast.TOD): instance of a TOD class.
        frame (int): the frame index.
        frame_offset (int): the first sample of the frame.
        frame_size (int): the number of samples in the the frame.
        frame_data (G3Frame): the input frame (only on rank zero).
        detector_map (str): the name of the frame timestream map.
        flag_map (str): then name of the frame flag map.

    Returns:
        None

    """
    comm = tod.mpicomm

    # First broadcast the frame data.
    if comm is not None:
        frame_data = comm.bcast(frame_data, root=0)

    # Local sample range
    local_first = tod.local_samples[0]
    nlocal = tod.local_samples[1]

    # Compute overlap of the frame with the local samples.
    cacheoff, froff, nfr = s3utils.local_frame_indices(
        local_first, nlocal, frame_offset, frame_size)

    # Helper function to actually copy a slice of data into cache.
    def copy_slice(data, fld, cache_prefix):
        cache_fld = fld
        if cache_prefix is not None:
            cache_fld = "{}{}".format(cache_prefix, fld)
        # Check field type and data shape
        ftype = s3utils.g3_dtype(data[fld])
        flen = len(data[fld])
        nnz = flen // frame_size
        if nnz * frame_size != flen:
            msg = "frame {}, field {} has length {} which is not "\
                "divisible by size {}".format(frame, fld, flen, frame_size)
            raise RuntimeError(msg)
        if not tod.cache.exists(cache_fld):
            # The field does not yet exist in cache, so create it.
            # print("proc {}:  create cache field {}, {}, ({},
            # {})".format(tod.mpicomm.rank, fld, ftype, tod.local_samples[1],
            # nnz), flush=True)
            if nnz == 1:
                rf = tod.cache.create(cache_fld, ftype,
                                      (tod.local_samples[1],))
            else:
                rf = tod.cache.create(cache_fld, ftype,
                                      (tod.local_samples[1], nnz))
            del rf
        # print("proc {}: get cache ref for {}".format(tod.mpicomm.rank,
        # cache_fld), flush=True)
        rf = tod.cache.reference(cache_fld)
        # Verify that the dimensions of the cache object are what we expect,
        # then copy the data.
        cache_samples = None
        cache_nnz = None
        if (len(rf.shape) > 1) and (rf.shape[1] > 0):
            # We have a packed 2D array
            cache_samples = rf.shape[0]
            cache_nnz = rf.shape[1]
        else:
            cache_nnz = 1
            cache_samples = len(rf)

        if cache_samples != tod.local_samples[1]:
            msg = "frame {}, field {}: cache has {} samples, which is"
            " different from local TOD size {}"\
                .format(frame, fld, cache_samples, tod.local_samples[1])
            raise RuntimeError(msg)

        if cache_nnz != nnz:
            msg = "frame {}, field {}: cache has nnz = {}, which is"\
                " different from frame nnz {}"\
                .format(frame, fld, cache_nnz, nnz)
            raise RuntimeError(msg)

        if cache_nnz > 1:
            slc = \
                np.array(data[fld][nnz*froff:nnz*(froff+nfr)],
                         copy=False).reshape((-1, nnz))
            # print("proc {}:  copy_slice field {}[{}:{},:] = \
            # frame[{}:{},:]".format(tod.mpicomm.rank, fld, cacheoff,
            # cacheoff+nfr, froff, froff+nfr), flush=True)
            rf[cacheoff:cacheoff+nfr, :] = slc
        else:
            slc = np.array(data[fld][froff:froff+nfr], copy=False)
            # print("proc {}:  copy_slice field {}[{}:{}] = \
            # frame[{}:{}]".format(tod.mpicomm.rank, fld, cacheoff,
            # cacheoff+nfr, froff, froff+nfr), flush=True)
            rf[cacheoff:cacheoff+nfr] = slc
        del rf
        return

    def copy_flags(chunks, fld, cache_prefix):
        ndata = np.zeros(froff+nfr, dtype=np.uint8)
        for beg, end in chunks.array():
            ndata[beg:end] = 1
        cache_fld = fld
        if cache_prefix is not None:
            cache_fld = "{}{}".format(cache_prefix, fld)
        # Check field type and data shape
        ftype = np.dtype(np.uint8)
        flen = len(ndata)
        nnz = flen // frame_size
        if nnz * frame_size != flen:
            msg = "frame {}, field {} has length {} which is not "\
                "divisible by size {}".format(frame, fld, flen, frame_size)
            raise RuntimeError(msg)
        if not tod.cache.exists(cache_fld):
            rf = tod.cache.create(cache_fld, ftype, (tod.local_samples[1],))
            del rf
        rf = tod.cache.reference(cache_fld)
        # Verify that the dimensions of the cache object are what we expect,
        # then copy the data.
        cache_samples = len(rf)

        if cache_samples != tod.local_samples[1]:
            msg = "frame {}, field {}: cache has {} samples, which is"
            " different from local TOD size {}"\
                .format(frame, fld, cache_samples, tod.local_samples[1])
            raise RuntimeError(msg)

        slc = np.array(ndata[froff:froff+nfr], copy=False)
        rf[cacheoff:cacheoff+nfr] = slc
        del rf
        return

    if cacheoff is not None:
        # print("proc {} has overlap with frame {}:  {} {} \
        # {}".format(tod.mpicomm.rank, frame, cacheoff, froff, nfr),
        # flush=True)

        # This process has some overlap with the frame.
        # FIXME:  need to account for multiple timestream maps.
        for field, fieldval in frame_data.iteritems():
            # Skip over maps
            if isinstance(fieldval, core3g.G3TimestreamMap):
                continue
            if isinstance(fieldval, core3g.G3MapVectorDouble):
                continue
            if isinstance(fieldval, core3g.G3MapVectorInt):
                continue
            if isinstance(fieldval, so3g.MapIntervalsInt):
                continue
            if isinstance(fieldval, so3g.IntervalsInt):
                copy_flags(fieldval, field, None)
            else:
                copy_slice(frame_data, field, None)

        dpats = None
        if (detector_map is not None) or (flag_map is not None):
            # Build our list of regex matches
            dpats = [re.compile(".*{}.*".format(d)) for d in tod.local_dets]

        if detector_map is not None:
            # If the field name contains any of our local detectors,
            # then cache it.
            for field in frame_data[detector_map].keys():
                for dp in dpats:
                    if dp.match(field) is not None:
                        # print("proc {} copy frame {}, field
                        # {}".format(tod.mpicomm.rank, frame, field),
                        # flush=True)
                        copy_slice(frame_data[detector_map], field, "signal_")
                        break
        if flag_map is not None:
            # If the field name contains any of our local detectors,
            # then cache it.
            for field in frame_data[flag_map].keys():
                for dp in dpats:
                    if dp.match(field) is not None:
                        chunks = frame_data[flag_map][field]
                        copy_flags(chunks, field, "flags_")
                        break
    return


def tod_to_frames(
        tod,
        start_frame,
        n_frames,
        frame_offsets,
        frame_sizes,
        cache_signal=None,
        cache_flags=None,
        cache_common_flags=None,
        copy_common=None,
        copy_detector=None,
        mask_flag_common=255,
        mask_flag=255,
        units=None,
        dets=None,
):
    """Gather all data from the distributed TOD cache for a set of frames.

    Args:
        tod (toast.TOD): instance of a TOD class.
        start_frame (int): the first frame index.
        n_frames (int): the number of frames.
        frame_offsets (array_like): list of the first samples of all frames.
        frame_sizes (list): list of the number of samples in each frame.
        cache_signal (str): if None, read signal from TOD.  Otherwise use this
            cache prefix for the detector signal timestreams.
        cache_flags (str): if None read det flags from TOD.  Otherwise use
            this cache prefix for the detector flag timestreams.
        cache_common_flags (str): if None, read common flags from TOD.
            Otherwise use this cache prefix.
        copy_common (tuple): (cache name, G3 type, frame name) of each extra
            common field to copy from cache.
        copy_detector (tuple): (cache name prefix, G3 type, G3 map type,
            frame name) of each distributed detector field (excluding the
            "signal") to copy from cache.
        mask_flag_common (int):  Bitmask to apply to common flags.
        mask_flag (int):  Bitmask to apply to per-detector flags.
        units: G3 units of the detector data.
        dets (list):  List of detectors to include in the frame.  If None,
            use all of the detectors in the TOD object.

    Returns:
        (list): List of frames on rank zero.  Other processes have a list of
            None values.

    """
    comm = tod.mpicomm
    rank = 0
    if comm is not None:
        rank = comm.rank
    comm_row = tod.grid_comm_row

    # Detector names
    if dets is None:
        detnames = tod.detectors
    else:
        detnames = []
        use_dets = set(dets)
        for det in tod.detectors:
            if det in use_dets:
                detnames.append(det)

    # Local sample range
    local_first = tod.local_samples[0]
    nlocal = tod.local_samples[1]

    # The process grid
    detranks, sampranks = tod.grid_size
    rankdet, ranksamp = tod.grid_ranks

    def get_local_cache(prow, fld, cacheoff, ncache):
        """Read a local slice of a cache field.
        """
        mtype = None
        pdata = None
        nnz = 0
        if rankdet == prow:
            ref = tod.cache.reference(fld)
            nnz = 1
            if (len(ref.shape) > 1) and (ref.shape[1] > 0):
                nnz = ref.shape[1]
            if ref.dtype == np.dtype(np.float64):
                mtype = MPI.DOUBLE
            elif ref.dtype == np.dtype(np.int64):
                mtype = MPI.INT64_T
            elif ref.dtype == np.dtype(np.int32):
                mtype = MPI.INT32_T
            elif ref.dtype == np.dtype(np.uint8):
                mtype = MPI.UINT8_T
            else:
                msg = "Cannot use cache field {} of type {}"\
                    .format(fld, ref.dtype)
                raise RuntimeError(msg)
            if cacheoff is not None:
                pdata = ref.flatten()[nnz*cacheoff:nnz*(cacheoff+ncache)]
            else:
                pdata = np.zeros(0, dtype=ref.dtype)
        return (pdata, nnz, mtype)

    def gather_field(prow, pdata, nnz, mpitype, cacheoff, ncache, tag):
        """Gather a single timestream buffer to the root process.
        """
        gdata = None
        # We are going to allreduce this later, so that every process
        # knows the dimensions of the field.
        gproc = 0
        allnnz = 0

        # Size of the local buffer
        pz = 0
        if pdata is not None:
            pz = len(pdata)

        if rankdet == prow:
            psizes = None
            if comm_row is None:
                psizes = [pz]
            else:
                psizes = comm_row.gather(pz, root=0)
            disp = None
            totsize = None
            if ranksamp == 0:
                # We are the process collecting the gathered data.
                allnnz = nnz
                gproc = rank
                # Compute the displacements into the receive buffer.
                disp = [0]
                for ps in psizes[:-1]:
                    last = disp[-1]
                    disp.append(last + ps)
                totsize = np.sum(psizes)
                # allocate receive buffer
                gdata = np.zeros(totsize, dtype=pdata.dtype)

            if comm_row is None:
                pdata[:] = gdata
            else:
                comm_row.Gatherv(pdata, [gdata, psizes, disp, mpitype], root=0)
            del disp
            del psizes

        # Now send this data to the root process of the whole communicator.
        # Only one process (the first one in process row "prow") has data
        # to send.

        if comm is not None:
            # All processes find out which one did the gather
            gproc = comm.allreduce(gproc, MPI.SUM)
            # All processes find out the field dimensions
            allnnz = comm.allreduce(allnnz, MPI.SUM)

        mtag = 10 * tag

        rdata = None
        if gproc == 0:
            if gdata is not None:
                if allnnz == 1:
                    rdata = gdata
                else:
                    rdata = gdata.reshape((-1, allnnz))
        else:
            # Data not yet on rank 0
            if rank == 0:
                # Receive data from the first process in this row
                rtype = comm.recv(source=gproc, tag=(mtag+1))
                rsize = comm.recv(source=gproc, tag=(mtag+2))
                rdata = np.zeros(rsize, dtype=np.dtype(rtype))
                comm.Recv(rdata, source=gproc, tag=mtag)
                # Reshape if needed
                if allnnz > 1:
                    rdata = rdata.reshape((-1, allnnz))
            elif (rank == gproc):
                # Send our data
                comm.send(gdata.dtype.char, dest=0, tag=(mtag+1))
                comm.send(len(gdata), dest=0, tag=(mtag+2))
                comm.Send(gdata, 0, tag=mtag)
        return rdata

    # For efficiency, we are going to gather the data for all frames at once.
    # Then we will split those up when doing the write.

    # Frame offsets relative to the memory buffers we are gathering
    fdataoff = [0]
    for f in frame_sizes[:-1]:
        last = fdataoff[-1]
        fdataoff.append(last+f)

    # The list of frames- only on the root process.
    fdata = None
    if rank == 0:
        fdata = [core3g.G3Frame(core3g.G3FrameType.Scan)
                 for f in range(n_frames)]
    else:
        fdata = [None for f in range(n_frames)]

    def flags_to_intervals(flgs):
        """Convert a flag vector to an interval list.
        """
        groups = [
            [i for i, value in it] for key, it in
            itertools.groupby(enumerate(flgs), key=operator.itemgetter(1))
            if key != 0]
        chunks = list()
        for grp in groups:
            chunks.append([grp[0], grp[-1]])
        return chunks

    def split_field(data, g3t, framefield, mapfield=None, g3units=units,
                    times=None):
        """Split a gathered data buffer into frames- only on root process.
        """
        if g3t == core3g.G3VectorTime:
            # Special case for time values stored as int64_t, but
            # wrapped in a class.
            for f in range(n_frames):
                dataoff = fdataoff[f]
                ndata = frame_sizes[f]
                g3times = list()
                for t in range(ndata):
                    g3times.append(core3g.G3Time(data[dataoff + t]))
                if mapfield is None:
                    fdata[f][framefield] = core3g.G3VectorTime(g3times)
                else:
                    fdata[f][framefield][mapfield] = \
                        core3g.G3VectorTime(g3times)
                del g3times
        elif g3t == so3g.IntervalsInt:
            # This means that the data is actually flags
            # and we should convert it into a list of intervals.
            fint = flags_to_intervals(data)
            for f in range(n_frames):
                dataoff = fdataoff[f]
                ndata = frame_sizes[f]
                datalast = dataoff + ndata
                chunks = list()
                idomain = (0, ndata-1)
                for intr in fint:
                    # Interval sample ranges are defined relative to the
                    # frame itself.
                    cfirst = None
                    clast = None
                    if (intr[0] < datalast) and (intr[1] >= dataoff):
                        # there is some overlap...
                        if intr[0] < dataoff:
                            cfirst = 0
                        else:
                            cfirst = intr[0] - dataoff
                        if intr[1] >= datalast:
                            clast = ndata - 1
                        else:
                            clast = intr[1] - dataoff
                        chunks.append([cfirst, clast])
                if mapfield is None:
                    if len(chunks) == 0:
                        fdata[f][framefield] = \
                            so3g.IntervalsInt()
                    else:
                        fdata[f][framefield] = \
                            so3g.IntervalsInt.from_array(
                                np.array(chunks, dtype=np.int64))
                    fdata[f][framefield].domain = idomain
                else:
                    if len(chunks) == 0:
                        fdata[f][framefield][mapfield] = \
                            so3g.IntervalsInt()
                    else:
                        fdata[f][framefield][mapfield] = \
                            so3g.IntervalsInt.from_array(
                                np.array(chunks, dtype=np.int64))
                        fdata[f][framefield][mapfield].domain = idomain
            del fint
        elif g3t == core3g.G3Timestream:
            if times is None:
                raise RuntimeError(
                    "You must provide the time stamp vector with a "
                    "Timestream object")
            for f in range(n_frames):
                dataoff = fdataoff[f]
                ndata = frame_sizes[f]
                if mapfield is None:
                    if g3units is None:
                        fdata[f][framefield] = \
                            g3t(data[dataoff:dataoff+ndata])
                    else:
                        fdata[f][framefield] = \
                            g3t(data[dataoff:dataoff+ndata], g3units)
                else:
                    if g3units is None:
                        fdata[f][framefield][mapfield] = \
                            g3t(data[dataoff:dataoff+ndata])
                    else:
                        fdata[f][framefield][mapfield] = \
                            g3t(data[dataoff:dataoff+ndata], g3units)
                timeslice = times[dataoff:dataoff + ndata]
                tstart = timeslice[0] * 1e8
                tstop = timeslice[-1] * 1e8
                fdata[f][framefield][mapfield].start = core3g.G3Time(tstart)
                fdata[f][framefield][mapfield].stop = core3g.G3Time(tstop)
        else:
            # The bindings of G3Vector seem to only work with
            # lists.  This is probably horribly inefficient.
            for f in range(n_frames):
                dataoff = fdataoff[f]
                ndata = frame_sizes[f]
                if len(data.shape) == 1:
                    fdata[f][framefield] = \
                        g3t(data[dataoff:dataoff+ndata].tolist())
                else:
                    # We have a 2D quantity
                    fdata[f][framefield] = \
                        g3t(data[dataoff:dataoff+ndata, :].flatten()
                            .tolist())
        return

    # Compute the overlap of all frames with the local process.  We want to
    # to find the full sample range that this process overlaps the total set
    # of frames.

    cacheoff = None
    ncache = 0

    for f in range(n_frames):
        # Compute overlap of the frame with the local samples.
        fcacheoff, froff, nfr = s3utils.local_frame_indices(
            local_first, nlocal, frame_offsets[f], frame_sizes[f])
        if fcacheoff is not None:
            if cacheoff is None:
                cacheoff = fcacheoff
                ncache = nfr
            else:
                ncache += nfr

    # Now gather the full sample data one field at a time.  The root process
    # splits up the results into frames.

    # First collect boresight data.  In addition to quaternions for the Az/El
    # pointing, we convert this back into angles that follow the specs
    # for telescope pointing.

    times = None
    if rankdet == 0:
        times = tod.local_times()
    if comm is not None:
        times = gather_field(0, times, 1, MPI.DOUBLE, cacheoff, ncache, 0)

    bore = None
    if rankdet == 0:
        bore = tod.read_boresight(local_start=cacheoff, n=ncache).flatten()
    if comm is not None:
        bore = gather_field(0, bore, 4, MPI.DOUBLE, cacheoff, ncache, 0)
    if rank == 0:
        split_field(bore.reshape(-1, 4), core3g.G3VectorDouble,
                    "qboresight_radec")

    bore = None
    if rankdet == 0:
        bore = tod.read_boresight_azel(
            local_start=cacheoff, n=ncache).flatten()
    if comm is not None:
        bore = gather_field(0, bore, 4, MPI.DOUBLE, cacheoff, ncache, 1)
    if rank == 0:
        split_field(bore.reshape(-1, 4), core3g.G3VectorDouble,
                    "qboresight_azel")

    if rank == 0:
        for f in range(n_frames):
            fdata[f]["boresight"] = core3g.G3TimestreamMap()
        ang_theta, ang_phi, ang_psi = qa.to_angles(bore)
        ang_az = ang_phi
        ang_el = (np.pi / 2.0) - ang_theta
        ang_roll = ang_psi
        split_field(ang_az, core3g.G3Timestream, "boresight", "az", None,
                    times=times)
        split_field(ang_el, core3g.G3Timestream, "boresight", "el", None,
                    times=times)
        split_field(ang_roll, core3g.G3Timestream, "boresight", "roll", None,
                    times=times)

    # Now the position and velocity information

    pos = None
    if rankdet == 0:
        pos = tod.read_position(local_start=cacheoff, n=ncache).flatten()
    if comm is not None:
        pos = gather_field(0, pos, 3, MPI.DOUBLE, cacheoff, ncache, 2)
    if rank == 0:
        split_field(pos.reshape(-1, 3), core3g.G3VectorDouble, "site_position")

    vel = None
    if rankdet == 0:
        vel = tod.read_velocity(local_start=cacheoff, n=ncache).flatten()
    if comm is not None:
        vel = gather_field(0, vel, 3, MPI.DOUBLE, cacheoff, ncache, 3)
    if rank == 0:
        split_field(vel.reshape(-1, 3), core3g.G3VectorDouble, "site_velocity")

    # Now handle the common flags- either from a cache object or from the
    # TOD methods

    cflags = None
    nnz = 1
    if cache_common_flags is None:
        if rankdet == 0:
            cflags = tod.read_common_flags(local_start=cacheoff, n=ncache)
            cflags &= mask_flag_common
    else:
        cflags, nnz, mtype = get_local_cache(0, cache_common_flags, cacheoff,
                                             ncache)
        if cflags is not None:
            cflags &= mask_flag_common
    if comm is not None:
        mtype = MPI.UINT8_T
        cflags = gather_field(0, cflags, nnz, mtype, cacheoff, ncache, 4)
    if rank == 0:
        split_field(cflags, so3g.IntervalsInt, "flags_common")

    # Any extra common fields

    if comm is not None:
        comm.barrier()

    if copy_common is not None:
        for cindx, (cname, g3typ, fname) in enumerate(copy_common):
            cdata, nnz, mtype = get_local_cache(0, cname, cacheoff, ncache)
            cdata = gather_field(0, cdata, nnz, mtype, cacheoff, ncache, cindx)
            if rank == 0:
                split_field(cdata, g3typ, fname)

    # Now read all per-detector quantities.

    # For each detector field, processes which have the detector
    # in their local_dets should be in the same process row.

    if rank == 0:
        for f in range(n_frames):
            fdata[f]["signal"] = core3g.G3TimestreamMap()
            fdata[f]["flags"] = so3g.MapIntervalsInt()
            if copy_detector is not None:
                for cname, g3typ, g3maptyp, fnm in copy_detector:
                    fdata[f][fnm] = g3maptyp()

    for dindx, dname in enumerate(detnames):
        drow = -1
        if dname in tod.local_dets:
            drow = rankdet
        # As a sanity check, verify that every process which
        # has this detector is in the same process row.
        rowcheck = None
        if comm is None:
            rowcheck = [drow]
        else:
            rowcheck = comm.gather(drow, root=0)
        prow = 0
        if rank == 0:
            rc = np.array([x for x in rowcheck if (x >= 0)],
                          dtype=np.int32)
            prow = np.max(rc)
            if np.min(rc) != prow:
                msg = "Processes with detector {} are not in the "\
                    "same row of the process grid\n".format(dname)
                sys.stderr.write(msg)
                if comm is not None:
                    comm.abort()

        # Every process finds out which process row is participating.
        if comm is not None:
            prow = comm.bcast(prow, root=0)

        # "signal"

        detdata = None
        nnz = 1
        if cache_signal is None:
            if rankdet == prow:
                detdata = tod.read(detector=dname, local_start=cacheoff,
                                   n=ncache)
        else:
            cache_det = "{}_{}".format(cache_signal, dname)
            detdata, nnz, mtype = get_local_cache(prow, cache_det, cacheoff,
                                                  ncache)
        if comm is not None:
            mtype = MPI.DOUBLE
            detdata = gather_field(prow, detdata, nnz, mtype, cacheoff,
                                   ncache, dindx)
        if rank == 0:
            split_field(detdata, core3g.G3Timestream, "signal",
                        mapfield=dname, times=times)

        # "flags"

        detdata = None
        nnz = 1
        if cache_flags is None:
            if rankdet == prow:
                detdata = tod.read_flags(detector=dname, local_start=cacheoff,
                                         n=ncache)
                detdata &= mask_flag
        else:
            cache_det = "{}_{}".format(cache_flags, dname)
            detdata, nnz, mtype = get_local_cache(prow, cache_det, cacheoff,
                                                  ncache)
            if detdata is not None:
                detdata &= mask_flag
        if comm is not None:
            mtype = MPI.UINT8_T
            detdata = gather_field(prow, detdata, nnz, mtype, cacheoff,
                                   ncache, dindx)
        if rank == 0:
            split_field(detdata, so3g.IntervalsInt, "flags", mapfield=dname)

        # Now copy any additional fields.

        if copy_detector is not None:
            for cname, g3typ, g3maptyp, fnm in copy_detector:
                cache_det = "{}_{}".format(cname, dname)
                detdata, nnz, mtype = get_local_cache(prow, cache_det,
                                                      cacheoff, ncache)
                detdata = gather_field(prow, detdata, nnz, mtype, cacheoff,
                                       ncache, dindx)
                if rank == 0:
                    split_field(detdata, g3typ, fnm, mapfield=dname,
                                times=times)

    return fdata
