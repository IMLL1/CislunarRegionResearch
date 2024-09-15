import numpy as np
from colorutils import hsv_to_hex
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


# NOTE: states is N x 6


# Dwell time at an altitude is inversely proportional to radial velocity; radial velocity is therefore an indicator
# This gives s/km; how long you're spending at an altitude per unit altitude
# to normalize, divide by period: gives periods per altitude
# Output is in s/km; how long you're spending in any km block
# MAJOR FLAW: Will give inf at apogee and perigee
def r_rate(states_rotating: list, periods: list, earth_x: float, ids: list):
    pos = [states[:, :3] for states in states_rotating]
    vel = [states[:, 3:] for states in states_rotating]
    pos = [pos_run - np.array([earth_x, 0, 0]) for pos_run in pos]
    r_rate = []
    radius = []
    for n in range(len(pos)):
        pos_run = pos[n]
        vel_run = vel[n]
        radius_run = np.linalg.vector_norm(pos_run, axis=1)
        radial_vel = (
            pos_run
            * np.sum(pos_run * vel_run, axis=1)[:, None]
            / radius_run[:, None] ** 2
        )
        radial_vel = np.linalg.vector_norm(radial_vel, axis=1)
        r_rate.append(np.abs(radial_vel * periods[n]))
        radius.append(radius_run)
    # return radius, dwell_time

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for idx in range(len(pos)):
        ax.plot(
            radius[idx],
            r_rate[idx],
            color=hsv_to_hex((360 * idx / len(pos), 0.75, 1)),
            label="id: " + str(ids[idx]),
        )
    ax.set_ylabel("Radial Rate [km/period]")
    ax.set_xlabel("Radius [km]")
    plt.title("Radial Rate vs Radius")
    plt.grid(linestyle="dashed", lw=0.5, c="gray")
    fig.legend()
    plt.show()


def hist_linear_density(
    states_rotating: list,
    times: list,
    periods: list,
    earth_x: float,
    ids: list,
    num_points: int = 1e7,
):
    num_points = int(num_points)
    pos = [
        states_run[:, :3] - np.array([earth_x, 0, 0]) for states_run in states_rotating
    ]
    radius = []
    for n in range(len(pos)):
        pos_run = pos[n]
        t = np.linspace(0, periods[n], num_points)
        pos_interp = np.array(
            [np.interp(t, times[n], pos_run[:, dim]) for dim in range(3)]
        ).T
        radius_run = np.linalg.vector_norm(pos_interp, axis=1)
        radius.append(radius_run)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for idx in range(len(pos)):
        y, x = np.histogram(
            radius[idx], bins=max([num_points // 5000, 100]), density=True
        )
        x = np.mean([x[1:], x[:-1]], axis=0)
        ax.plot(
            x,
            y,
            color=hsv_to_hex((360 * idx / len(pos), 0.75, 1)),
            label="id: " + str(ids[idx]),
        )
    ax.set_ylabel("Probability Density")
    ax.set_xlabel("Radius [km]")
    plt.title("Probability vs Radius")
    plt.grid(linestyle="dashed", lw=0.5, c="gray")
    fig.legend()
    plt.show()
    
    return y, x


def kern_linear_density(
    states_rotating: list,
    times: list,
    periods: list,
    earth_x: float,
    ids: list,
    num_points: int = 1e5,
):
    num_points = int(num_points)
    pos = [
        states_run[:, :3] - np.array([earth_x, 0, 0]) for states_run in states_rotating
    ]
    radius = []
    pdfs = []
    for n in range(len(pos)):
        pos_run = pos[n]
        t = np.linspace(0, periods[n], num_points)
        pos_interp = np.array(
            [np.interp(t, times[n], pos_run[:, dim]) for dim in range(3)]
        ).T
        radius_run = np.linalg.vector_norm(pos_interp, axis=1)
        radius.append(radius_run)
        pdfs.append(gaussian_kde(radius_run))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for idx in range(n):
        radii = np.linspace(min(radius[idx]), max(radius[idx]), int(5e2))
        pdf_val = pdfs[idx].pdf(radii)
        ax.plot(
            radii,
            pdf_val,
            color=hsv_to_hex((360 * idx / len(pos), 0.75, 1)),
            label="id: " + str(ids[idx]),
        )
    ax.set_ylabel("Probability Density")
    ax.set_xlabel("Radius [km]")
    plt.title("Probability vs Radius")
    plt.grid(linestyle="dashed", lw=0.5, c="gray")
    fig.legend()
    plt.show()


def altitude_blocks(
    states_rotating: list,
    times: list,
    periods: list,
    earth_x: float,
    ids: list,
    blocks: dict = {
        "LEO": (6371 + 100, 6371 + 1500),
        "MEO": (6371 + 15000, 6371 + 25000),
        "GEO": (42000, 42200),
    },
):
    # find times that it passes each point
    radius = [
        np.linalg.vector_norm(states_run[:, :3] - np.array([earth_x, 0, 0]), axis=1)
        for states_run in states_rotating
    ]
    dwelltimes = []

    for n in range(len(radius)):
        radius_run = radius[n]
        dwelltimes.append({})
        for block in blocks.keys():
            lb, ub = blocks[block]
            lb_crossings = np.diff(np.sign(radius_run - lb))
            ub_crossings = np.diff(np.sign(radius_run - ub))
            # PSUEDO CODE:
            # 1. coarsely locate crossings of both lower and upper bounds
            # 2. if first crossing is exit, add the period to it (TBD how)
            # 3. interpolate to find fine precision crossing times
            # 4. pair up crossings, find duration of each pair
            # 5. sum of durations, divide by period
            # coarse location of crossings of both lower and upper bound

            # 1. coarsely locate crossings of both lower and upper bounds
            lb_guess = np.nonzero(lb_crossings)[0]
            ub_guess = np.nonzero(ub_crossings)[0]
            crossings_idx = np.sort([*lb_guess, *ub_guess])

            # 2. if first crossing is exit, move it
            if crossings_idx[0] in lb_guess:
                direction = lb_crossings[crossings_idx[0]]
            else:
                direction = ub_crossings[crossings_idx[0]]
            if direction < 0:
                crossings_idx = np.array([*crossings_idx[1:], crossings_idx[0]])
                toAdd = periods[n]
            else:
                toAdd = 0

            assert np.mod(len(crossings_idx), 2) == 0

            crossing_times = 0.0 * crossings_idx
            # 3: interpolate to find precise crossing times
            for xnum in range(len(crossings_idx)):
                guess = crossings_idx[xnum]  # guess index
                # upper or lower bound, depending on which way we're crossing
                bound = lb if lb_crossings[guess] else ub
                # linear interpolation
                tcross = times[n][guess] - (radius_run - bound)[guess] * (
                    times[n][guess + 1] - times[n][guess]
                ) / (radius_run[guess + 1] - radius_run[guess])
                crossing_times[xnum] = tcross
                # if this is the last crossing AND it's an exit, add the period to it so that we exit at the end instead of the beginning
                if xnum == len(crossings_idx) - 1:
                    crossing_times[xnum] += toAdd

            # 4. pair up crossings, find duration of each pair
            dwelltimes[n][block] = 0
            for xnum in range(len(crossing_times) // 2):
                dwelltimes[n][block] += (
                    crossing_times[2 * xnum + 1] - crossing_times[2 * xnum]
                )
            dwelltimes[n][block] /= periods[n]

            # if no crossings, see if dwelltimes is 0 or 1
            if radius_run[0] >= lb and radius_run[0] <= ub:
                dwelltimes[n][block] = 1

    return dwelltimes
