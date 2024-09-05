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
        radial_vel = pos_run * np.sum(pos_run * vel_run) / radius_run[:, None] ** 2
        r_rate.append(np.abs(radial_vel / periods[n]))
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
    ax.set_ylabel("Dwell Time [km/period]")
    ax.set_xlabel("Radius [km]")
    plt.title("Dwell Time vs Radius")
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
        binwidth = np.mean(np.diff(x))
        y *= binwidth
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


def kern_linear_density(
    states_rotating: list,
    times: list,
    periods: list,
    earth_x: float,
    ids: list,
    num_points: int = 1e6,
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
