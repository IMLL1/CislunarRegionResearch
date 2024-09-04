from CR3BP_propagator import *
from colorutils import hsv_to_hex
from read_ICs import read_ICs
np.random.seed(0)

ids = [25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
tf, DRO_ICs = read_ICs(ids)
base_propagator = CR3BP(LU=389703, TU=382981)

inertial_states = []
CR3BP_states = []
for idx in range(len(ids)):
    base_propagator.propagate(DRO_ICs[idx,:], tf[idx])
    base_propagator.get_inertial_states()
    inertial = base_propagator.states[:,:3]/6371
    cr3bp = base_propagator.states_cr3bp[:,:3]/6371
    inertial_states.append(inertial)
    CR3BP_states.append(cr3bp)

inertial_lines = []
fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")
ax.plot(0,0,0, lw=0)
for idx in range(len(ids)):
    line, = ax.plot(inertial_states[idx][:, 0], inertial_states[idx][:, 1], inertial_states[idx][:, 2], lw=1, color=hsv_to_hex((360*idx/len(ids), 0.75,1)))
    inertial_lines.append(line)
ax.plot(base_propagator.moonstate[:, 0]/6371, base_propagator.moonstate[:, 1]/6371, base_propagator.moonstate[:, 2]/6371, lw=0.5, color='grey')
plt.axis("equal")
ax.set_xlabel("ECI x [ER]")
ax.set_ylabel("ECI y [ER]")
ax.set_zlabel("ECI z [ER]")
plt.title("Trajectory")
plt.grid(linestyle="dashed", lw=0.5, c="gray")

CR3BP_lines = []
ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter((1-base_propagator.mu)*base_propagator.LU/3671,0,0, s=2, color="black")
for idx in range(len(ids)):
    line, = ax2.plot(CR3BP_states[idx][:, 0], CR3BP_states[idx][:, 1],CR3BP_states[idx][:, 2], lw=1, c=hsv_to_hex((360*idx/len(ids), 0.75,1)))
    CR3BP_lines.append(line)
ax2.scatter((-base_propagator.mu)*base_propagator.LU/3671,0,0, s=3, color="black")

plt.axis("equal")
ax2.set_xlabel("ECR x [ER]")
ax2.set_ylabel("ECR y [ER]")
ax2.set_zlabel("ECR z [ER]")
plt.title("Trajectory (CR3BP Coordinates)")
plt.grid(linestyle="dashed", lw=0.5, c="gray")
fig.legend(tuple(inertial_lines), ("ID: " + str(id) for id in ids), loc="lower center", ncol=len(ids))
plt.show()