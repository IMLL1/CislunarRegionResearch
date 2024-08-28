from CR3BP_propagator import *
np.random.seed(0)

# DROs most interesting

DRO_IC = [3.0680940168765748E-2,	1.0441909756057249E-26,	-5.4104412657613781E-31,	-1.9798683479304902E-12,	6.6762053214726942E+0,	2.4008450093736843E-29]
tf = 6.5*2

base_propagator = CR3BP(LU=389703, TU=382981)
base_propagator.propagate(DRO_IC, tf)
base_propagator.get_inertial_states()
base_inertial = base_propagator.states[:,:3]/6371
base_CR3BP = base_propagator.states_cr3bp[:,:3]/6371
num_MCs = 20

# 1 km is around 2.5e-6. The velocity is pretty much 1-to-1. 1e-3 is a meter, 1e-5 is a cm
dX_sigma = [1e-9, 1e-9, 1e-9, 1e-8, 1e-7, 1e-8]
dX_sigma = [0, 0, 0, 1e-5, 1e-5, 1e-5]

MC_ICs = np.random.multivariate_normal(DRO_IC, np.diag(dX_sigma), num_MCs)
MC_inertial = []
MC_CR3BP = []
for idx in range(num_MCs):
    base_propagator.propagate(MC_ICs[idx,:], tf)
    base_propagator.get_inertial_states()
    inertial = base_propagator.states[:,:3]/6371
    cr3bp = base_propagator.states_cr3bp[:,:3]/6371
    MC_inertial.append(inertial)
    MC_CR3BP.append(cr3bp)


fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")
ax.plot(0,0,0, lw=0)
for idx in range(num_MCs):
    ax.plot(MC_inertial[idx][:, 0], MC_inertial[idx][:, 1], MC_inertial[idx][:, 2], lw=1, c='red')
ax.plot(base_inertial[:, 0], base_inertial[:, 1], base_inertial[:, 2], lw=1, c='black')
ax.plot(base_propagator.moonstate[:, 0]/6371, base_propagator.moonstate[:, 1]/6371, base_propagator.moonstate[:, 2]/6371, lw=0.5, c='grey')
plt.axis("equal")
ax.set_xlabel("ECI x [ER]")
ax.set_ylabel("ECI y [ER]")
ax.set_zlabel("ECI z [ER]")
plt.title("Trajectory")
plt.grid(linestyle="dashed", lw=0.5, c="gray")

ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter((1-base_propagator.mu)*base_propagator.LU/3671,0,0, s=2, c="black")
ax2.scatter((-base_propagator.mu)*base_propagator.LU/3671,0,0, s=3, c="black")
for idx in range(num_MCs):
    ax2.plot(MC_CR3BP[idx][:, 0], MC_CR3BP[idx][:, 1],MC_CR3BP[idx][:, 2], lw=1, c='red')
ax2.plot(base_CR3BP[:, 0], base_CR3BP[:, 1],base_CR3BP[:, 2], lw=1, c='black')
plt.axis("equal")
ax2.set_xlabel("ECR x [ER]")
ax2.set_ylabel("ECR y [ER]")
ax2.set_zlabel("ECR z [ER]")
plt.title("Trajectory (CR3BP Coordinates)")
plt.grid(linestyle="dashed", lw=0.5, c="gray")
plt.show()