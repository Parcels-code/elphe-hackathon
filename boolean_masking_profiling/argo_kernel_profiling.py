import argparse
import cProfile
from pathlib import Path
import pstats

import xarray as xr
import numpy as np

import parcels


def profile_execution_time(argo_kernel: str):
    report_name = f"profiles/time/argo_{argo_kernel.__name__}.prof"
    Path(report_name).parent.mkdir(parents=True, exist_ok=True)
    prof = cProfile.Profile()
    prof.enable()
    run_simulation(argo_kernel)
    prof.disable()
    stats = pstats.Stats(prof)
    stats.print_stats(
        r"run_simulation|particleset\.py:\d+\(execute\)|ArgoVerticalMovement"
    )
    stats.dump_stats(report_name)


# Define the new Kernel that mimics Argo vertical movement
driftdepth = 1000  # maximum depth in m
mindepth = 1.0  # minimum depth in m
maxdepth = 2000  # maximum depth in m
vertical_speed = 0.10  # sink and rise speed in m/s

# Setting shorter cycle times for testing purposes
cycletime = 3 * 86400  # total time of cycle in seconds
drifttime = 2 * 86400  # time of deep drift in seconds


def ArgoVerticalMovement_npwhere(particles, fieldset):
    # Split particles based on their current cycle_phase
    ptcls0 = particles[particles.cycle_phase == 0]
    ptcls1 = particles[particles.cycle_phase == 1]
    ptcls2 = particles[particles.cycle_phase == 2]
    ptcls3 = particles[particles.cycle_phase == 3]
    ptcls4 = particles[particles.cycle_phase == 4]

    # Phase 0: Sinking with vertical_speed until depth is driftdepth
    ptcls0.dz += vertical_speed * ptcls0.dt
    ptcls0.cycle_phase = np.where(
        ptcls0.z + ptcls0.dz >= driftdepth, 1, ptcls0.cycle_phase
    )
    ptcls0.dz = np.where(
        ptcls0.z + ptcls0.dz >= driftdepth, driftdepth - ptcls0.z, ptcls0.dz
    )

    # Phase 1: Drifting at depth for drifttime seconds
    ptcls1.drift_age += ptcls1.dt
    ptcls1.cycle_phase = np.where(ptcls1.drift_age >= drifttime, 2, ptcls1.cycle_phase)
    ptcls1.drift_age = np.where(ptcls1.drift_age >= drifttime, 0, ptcls1.drift_age)

    # Phase 2: Sinking further to maxdepth
    ptcls2.dz += vertical_speed * ptcls2.dt
    ptcls2.cycle_phase = np.where(
        ptcls2.z + ptcls2.dz >= maxdepth, 3, ptcls2.cycle_phase
    )
    ptcls2.dz = np.where(
        ptcls2.z + ptcls2.dz >= maxdepth, maxdepth - ptcls2.z, ptcls2.dz
    )

    # Phase 3: Rising with vertical_speed until at surface
    ptcls3.dz -= vertical_speed * ptcls3.dt
    # ptcls3.temp = fieldset.thetao[ptcls3.time, ptcls3.z, ptcls3.lat, ptcls3.lon]  # thetao not available
    ptcls3.cycle_phase = np.where(
        ptcls3.z + ptcls3.dz <= mindepth, 4, ptcls3.cycle_phase
    )
    ptcls3.dz = np.where(
        ptcls3.z + ptcls3.dz <= mindepth,
        mindepth - ptcls3.z,
        ptcls3.dz,
    )

    # Phase 4: Transmitting at surface until cycletime is reached
    ptcls4.cycle_phase = np.where(ptcls4.cycle_age >= cycletime, 0, ptcls4.cycle_phase)
    ptcls4.cycle_age = np.where(ptcls4.cycle_age >= cycletime, 0, ptcls4.cycle_age)
    ptcls4.temp = np.nan  # no temperature measurement when at surface

    particles.cycle_age += particles.dt  # update cycle_age


def ArgoVerticalMovement_boolean(particles, fieldset):
    # Split particles based on their current cycle_phase
    ptcls0 = particles[particles.cycle_phase == 0]
    ptcls1 = particles[particles.cycle_phase == 1]
    ptcls2 = particles[particles.cycle_phase == 2]
    ptcls3 = particles[particles.cycle_phase == 3]
    ptcls4 = particles[particles.cycle_phase == 4]

    # Phase 0: Sinking with vertical_speed until depth is driftdepth
    ptcls0.dz += vertical_speed * ptcls0.dt
    next_phase = ptcls0.z + ptcls0.dz >= driftdepth
    ptcls0.cycle_phase[next_phase] = 1
    ptcls0.dz[next_phase] = driftdepth - ptcls0.z[next_phase]  # avoid overshoot

    # Phase 1: Drifting at depth for drifttime seconds
    ptcls1.drift_age += ptcls1.dt
    next_phase = ptcls1.drift_age >= drifttime
    ptcls1.cycle_phase[next_phase] = 2
    ptcls1.drift_age[next_phase] = 0  # reset drift_age for next cycle

    # Phase 2: Sinking further to maxdepth
    ptcls2.dz += vertical_speed * ptcls2.dt
    next_phase = ptcls2.z + ptcls2.dz >= maxdepth
    ptcls2.cycle_phase[next_phase] = 3
    ptcls2.dz[next_phase] = maxdepth - ptcls2.z[next_phase]  # avoid overshoot

    # Phase 3: Rising with vertical_speed until at surface
    ptcls3.dz -= vertical_speed * ptcls3.dt
    # ptcls3.temp = fieldset.thetao[ptcls3.time, ptcls3.z, ptcls3.lat, ptcls3.lon]  # thetao not available
    next_phase = ptcls3.z + ptcls3.dz <= mindepth
    ptcls3.cycle_phase[next_phase] = 4
    ptcls3.dz[next_phase] = mindepth - ptcls3.z[next_phase]  # avoid overshoot

    # Phase 4: Transmitting at surface until cycletime is reached
    next_phase = ptcls4.cycle_age >= cycletime
    ptcls4.cycle_phase[next_phase] = 0
    ptcls4.cycle_age[next_phase] = 0  # reset cycle_age for next cycle
    ptcls4.temp = np.nan  # no temperature measurement when at surface

    particles.cycle_age += particles.dt  # update cycle_age


def run_simulation(argo_kernel: str) -> None:
    filename = "physics_compressed.zarr"

    ds = xr.open_zarr(filename)
    ds.load()
    fieldset = parcels.FieldSet.from_sgrid_conventions(ds, mesh="spherical")

    ArgoParticle = parcels.Particle.add_variable(
        [
            parcels.Variable("cycle_phase", dtype=np.int32, initial=0.0),
            parcels.Variable("cycle_age", dtype=np.float32, initial=0.0),
            parcels.Variable("drift_age", dtype=np.float32, initial=0.0),
            parcels.Variable("temp", dtype=np.float32, initial=np.nan),
        ]
    )

    N = 10_000
    X, Y = np.meshgrid(
        np.linspace(-80, 10, int(np.sqrt(N))), np.linspace(-15, 35, int(np.sqrt(N)))
    )

    pset = parcels.ParticleSet(
        fieldset=fieldset,
        pclass=ArgoParticle,
        lon=X,
        lat=Y,
        z=10 * np.ones_like(X),
    )
    pfile = parcels.ParticleFile(
        "output_argo.parquet",
        outputdt=np.timedelta64(2, "h"),
        mode="w",
    )
    kernels = [argo_kernel, parcels.kernels.AdvectionRK2]

    pset.execute(
        kernels,
        runtime=np.timedelta64(6, "D"),
        dt=np.timedelta64(15, "m"),
        output_file=pfile,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--argo-kernel",
        choices=[
            "npwhere",
            "boolean",
        ],
        default="npwhere",
        help="Which Argo kernel to use for the simulation.",
    )
    args = parser.parse_args()

    argo_kernel = (
        ArgoVerticalMovement_npwhere
        if args.argo_kernel == "npwhere"
        else ArgoVerticalMovement_boolean
    )
    profile_execution_time(argo_kernel)
