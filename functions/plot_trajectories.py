from colorutils import hsv_to_hex
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def plot_trajectories(
    inertial_states: list,
    CR3BP_states: list,
    moonstates: NDArray,
    bodies_x: list,
    ids: list,
):
    n = len(inertial_states)
    inertial_lines = []
    fig = plt.figure()
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(0, 0, 0, lw=0)
    for idx in range(n):
        (line,) = ax.plot(
            inertial_states[idx][:, 0],
            inertial_states[idx][:, 1],
            inertial_states[idx][:, 2],
            lw=1,
            color=hsv_to_hex((360 * idx / n, 0.75, 1)),
            label="id: " + str(ids[idx]),
        )
        inertial_lines.append(line)
    ax.plot(moonstates[:, 0], moonstates[:, 1], moonstates[:, 2], lw=0.5, color="grey")
    plt.axis("equal")
    ax.set_xlabel("ECI x [km]")
    ax.set_ylabel("ECI y [km]")
    ax.set_zlabel("ECI z [km]")
    plt.title("Trajectory (Inertial)")
    plt.grid(linestyle="dashed", lw=0.5, c="gray")

    CR3BP_lines = []
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(bodies_x[0], 0, 0, s=2, color="black")
    for idx in range(n):
        (line,) = ax2.plot(
            CR3BP_states[idx][:, 0],
            CR3BP_states[idx][:, 1],
            CR3BP_states[idx][:, 2],
            lw=1,
            c=hsv_to_hex((360 * idx / n, 0.75, 1)),
        )
        CR3BP_lines.append(line)
    ax2.scatter(bodies_x[1], 0, 0, s=3, color="black")

    plt.axis("equal")
    ax2.set_xlabel("ECR x [km]")
    ax2.set_ylabel("ECR y [km]")
    ax2.set_zlabel("ECR z [km]")
    plt.title("Trajectory (EM Rotating)")
    plt.grid(linestyle="dashed", lw=0.5, c="gray")
    fig.legend(loc="lower center", ncol=n)
    plt.show()
