from CR3BP import CR3BP
import numpy as np

prop = CR3BP(odemethod="LSODA")
# x0 = np.array([0.824293579889, 0, 0.05998471178, 0, 0.17085057273180781, 0])
# tf = 1.9810382309

x0 = np.array([9.0998753221126e-1, 0, 1.48301191319791e-1, 0, -2.91883100990976e-2, 0])
tf = 3.3852875941088589

# x0 = np.array([0.8313996, 0, 0.12248709, 0, 0.2370638, 0])
# tf = 2.784659402929

# x0 = np.array([0.93073122691776, 0, 0.2306551858505, 0, 0.10356396251875, 0])
# tf = 1.84245395871

optstate = prop.find_periodic_orbit(
    opt_vars=["tf", "z", "vy"],
    obj_zero=["y", "vx"],
    init_guess=[tf / 2, *x0],
    tol=3e-12,
)
x0half = optstate[1:]
tfhalf = optstate[0]
print("Half-period found")
optstate = prop.find_periodic_orbit(
    opt_vars=["tf", "z", "vy"],
    obj_zero=["y", "vx"],
    init_guess=[2 * tfhalf, *x0half],
    tol=3e-12,
)
x0 = optstate[1:]
tf = optstate[0]
print("Periodic orbit found\n")

x0 = optstate[1:]
tf = optstate[0]

xs = prop.propagate(x0, tf, tol=3e-16)
print("Trajectory propagated")

npts = 50
xs_s1, xs_s2, xs_u1, xs_u2 = prop.manifold_curves(
    x0=x0, period=tf, npts=npts, d=3e-4, tol=1e-10
)  # , termtime=15
# )
print("manifold propagated")

import matplotlib.pyplot as plt

ax = plt.figure(figsize=(6, 6)).add_subplot(projection="3d")
x, y, z, vx, vy, vz = xs.T
for i in range(npts):
    plt.plot(xs_s1[i][:, 0], xs_s1[i][:, 1], xs_s1[i][:, 2], "-b", lw=0.2)
    plt.plot(xs_s2[i][:, 0], xs_s2[i][:, 1], xs_s2[i][:, 2], "-c", lw=0.2)
    plt.plot(xs_u1[i][:, 0], xs_u1[i][:, 1], xs_u1[i][:, 2], "-r", lw=0.2)
    plt.plot(xs_u2[i][:, 0], xs_u2[i][:, 1], xs_u2[i][:, 2], "-m", lw=0.2)

plt.plot(1 - prop.mu, 0, 0, ".", color="dimgray", ms=7.5, label="Moon")
plt.plot(-prop.mu, 0, 0, ".", color="green", ms=15, label="Earth")
plt.plot(x, y, z, "-k", label="Periodic Orbit")
plt.plot(np.nan, np.nan, np.nan, "-b", label="Stable Half 1")
plt.plot(np.nan, np.nan, np.nan, "-c", label="Stable Half 2")
plt.plot(np.nan, np.nan, np.nan, "-r", label="Unstable Half 1")
plt.plot(np.nan, np.nan, np.nan, "-m", label="Unstable Half 2")
plt.plot(x, y, z, "-k")
plt.axis("equal")
plt.title("CR3BP Manifolds")
ax.set_xlabel("$x$ [n.d.]")
ax.set_ylabel("$y$ [n.d.]")
ax.set_zlabel("$z$ [n.d.]")
plt.legend(ncols=2)
plt.show()
