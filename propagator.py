from CR3BP_propagator import *

# DROs most interesting

DRO_IC = [3.0680940168765748E-2,	1.0441909756057249E-26,	-5.4104412657613781E-31,	-1.9798683479304902E-12,	6.6762053214726942E+0,	2.4008450093736843E-29]

base_CR3BP = CR3BP(LU=389703, TU=382981)

r_sigma = base_CR3BP.LU * 0.1 # 100 meters
v_sigma = r_sigma / 100 # 1 m.s



base_CR3BP.propagate(DRO_IC, 6.3)
base_CR3BP.get_inertial_states()

pos_earthrad = base_CR3BP.states[:,:3]/6371

fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")
ax.plot(0,0,0, lw=0)
ax.plot(pos_earthrad[:, 0], pos_earthrad[:, 1], pos_earthrad[:, 2], lw=1, c='red')
ax.plot(base_CR3BP.moonstate[:, 0]/6371, base_CR3BP.moonstate[:, 1]/6371, base_CR3BP.moonstate[:, 2]/6371, lw=0.5, c='grey')
plt.axis("equal")
ax.set_xlabel("ECI x [ER]")
ax.set_ylabel("ECI y [ER]")
ax.set_zlabel("ECI z [ER]")
plt.title("Trajectory (3D)")
plt.grid(linestyle="dashed", lw=0.5, c="gray")

ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter((1-base_CR3BP.mu)*base_CR3BP.LU/3671,0,0, s=2, c="black")
ax2.scatter((-base_CR3BP.mu)*base_CR3BP.LU/3671,0,0, s=3, c="black")
ax2.plot(base_CR3BP.states_cr3bp[:, 0]/6371, base_CR3BP.states_cr3bp[:, 1]/6371,base_CR3BP.states_cr3bp[:, 2]/6371, lw=1, c='red')
plt.axis("equal")
ax2.set_xlabel("ECR x [ER]")
ax2.set_ylabel("ECR y [ER]")
ax2.set_zlabel("ECR z [ER]")
plt.title("Trajectory (3D CR3BP)")
plt.grid(linestyle="dashed", lw=0.5, c="gray")
plt.show()