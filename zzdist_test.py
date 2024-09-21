import matplotlib.pyplot as plt
from colorutils import hsv_to_hex


import numpy as np
from functions.propagators import CR3BP

plt.style.use("dark_background")

obj = CR3BP()
tf = 6.179
obj.propagate(
    [
        0.3618718523685345,
        9.218123740185406e-24,
        2.7335981255904526e-24,
        1.2868252128805022e-12,
        1.7090727637352776,
        1.458418206338232e-23,
    ],
    tf,
    np.linspace(0, tf, 250),
)
states = obj.states_cr3bp
ts = obj.ts_nd
bodies_x = (-obj.mu, 1 - obj.mu)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
CR3BP_lines = []
ax.scatter(bodies_x[0], 0, 0, s=2, color="black")
ax.scatter(
    states[:, 0],
    states[:, 1],
    states[:, 2],
    c=np.linalg.vector_norm(states, axis=1),
    s=1
)
ax.scatter(bodies_x[1], 0, 0, s=3, color="black")

plt.axis("equal")
ax.set_xlabel("ECR x [km]")
ax.set_ylabel("ECR y [km]")
ax.set_zlabel("ECR z [km]")
plt.title("Trajectory (EM Rotating)")
plt.grid(linestyle="dashed", lw=0.5, c="gray")
plt.show()
