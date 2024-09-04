import numpy as np
from colorutils import hsv_to_hex

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
        r_rate.append(np.abs(radial_vel/periods[n]))
        radius.append(radius_run)
    # return radius, dwell_time
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    lines = []
    for idx in range(n):
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
