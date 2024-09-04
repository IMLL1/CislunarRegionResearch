from CR3BP_propagator import CR3BP
from read_ICs import read_ICs
from plot_trajectories import plot_trajectories
from numpy.random import seed
from dwell_times import r_rate

seed(0)

# ids = [25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
ids = [10, 25, 100, 500, 1000]
tf, DRO_ICs = read_ICs(ids)
base_propagator = CR3BP(LU=389703, TU=382981)

inertial_states = []
CR3BP_states = []
for idx in range(len(ids)):
    base_propagator.propagate(DRO_ICs[idx, :], tf[idx])
    base_propagator.get_inertial_states()
    inertial = base_propagator.states
    cr3bp = base_propagator.states_cr3bp
    inertial_states.append(inertial)
    CR3BP_states.append(cr3bp)

r_rate(
    CR3BP_states,
    tf,
    (-base_propagator.mu) * base_propagator.LU,
    ids,
)
# dwell_times = inv_vel()

plot_trajectories(
    inertial_states,
    CR3BP_states,
    base_propagator.moonstate,
    [
        (1 - base_propagator.mu) * base_propagator.LU,
        (-base_propagator.mu) * base_propagator.LU,
    ],
    ids,
)
