# Copyright (c) 2020 Simons Observatory.
# Full license can be found in the top level "LICENSE" file.

"""Check functionality of the wiregrid simulation

"""

import unittest
import numpy as np

import astropy.units as u

try:
    import toast
    import sotodlib.toast as sotoast
    import sotodlib.toast.ops as so_ops
    toast_available = True
except ImportError as e:
    toast_available = False


class SimWireGridTest(unittest.TestCase):
    def test_instantiate(self):
        """Test instantiating a simulation operator."""
        if not toast_available:
            print("toast cannot be imported- skipping unit tests", flush=True)
            return

        comm, procs, rank = toast.get_world()
        toast_comm = toast.Comm(world=comm, groupsize=procs)
        data = toast.Data(comm=toast_comm)

        focalplane = sotoast.SOFocalplane(
            hwfile=None,
            telescope="SAT1",
            sample_rate=10 * u.Hz,
            bands="SAT_f090",
            wafer_slots=None,
            tube_slots=None,
            thinfp=16,
            comm=comm,
        )

        fname_schedule = "schedule.txt"
        el = 45
        name = "wiregrid_calibration"
        az = 180
        with open(fname_schedule, "w") as handle:
            handle.write(
                "ATACAMA SAT1 -22.958 -67.786 5200.0\n"
            )
            handle.write(
                f" 2025-01-01 00:00:00  2025-01-01 00:10:00 "
                f"0.0 0.0 0.0 "
                f"{name} {az} {az} {el} "
                f"S 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 "
                f"0 0\n"
            )
        schedule = toast.schedule.GroundSchedule()
        schedule.read(fname_schedule, comm=comm)

        site = toast.instrument.GroundSite(
            schedule.site_name,
            schedule.site_lat,
            schedule.site_lon,
            schedule.site_alt,
            weather=None,
        )
        telescope = toast.instrument.Telescope(
            schedule.telescope_name, focalplane=focalplane, site=site
        )

        sim_ground = toast.ops.SimGround(
            name="sim_ground",
            weather="atacama",
            detset_key="pixel",
            telescope=telescope,
            schedule=schedule,
        )
        sim_ground.apply(data)

        pointing = toast.ops.PointingDetectorSimple(
            name="det_pointing_azel",
            quats="quats_azel",
            boresight=sim_ground.boresight_azel,
        )
        weights = toast.ops.StokesWeights(
            name="weights_azel",
            weights="weights_azel",
            mode="IQU",
            detector_pointing=pointing,
        )

        so_ops.SimWireGrid(
            name="sim_wiregrid",
            detector_pointing=pointing,
            detector_weights=weights,
            wiregrid_angular_speed=10.0*u.degree/u.second,
            wiregrid_angular_acceleration=0.1*u.degree/u.second**2,
        ).apply(data)

if __name__ == '__main__':
    unittest.main()
