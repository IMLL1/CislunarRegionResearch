from propagators import CR3BP
from read_ICs import read_ICs
from plot_trajectories import plot_trajectories

# from numpy.random import seed
# seed(0)
from dwell_times import r_rate, hist_linear_density, kern_linear_density

ids = [25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600]
# ids = [1075, 1100, 1125, 1140]
tf, DRO_ICs = read_ICs(ids)
base_propagator = CR3BP(LU=389703, TU=382981)

inertial_states = []
CR3BP_states = []
times = []
for idx in range(len(ids)):
    base_propagator.propagate(DRO_ICs[idx, :], tf[idx])
    base_propagator.get_inertial_states()
    inertial_states.append(base_propagator.states)
    CR3BP_states.append(base_propagator.states_cr3bp)
    times.append(base_propagator.ts)

# Dwell Time functions

r_rate(
    CR3BP_states,
    tf,
    (-base_propagator.mu) * base_propagator.LU,
    ids,
)

hist_linear_density(
    CR3BP_states,
    times,
    tf * base_propagator.TU,
    (-base_propagator.mu) * base_propagator.LU,
    ids,
)

kern_linear_density(
    CR3BP_states,
    times,
    tf * base_propagator.TU,
    (-base_propagator.mu) * base_propagator.LU,
    ids,
)

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
