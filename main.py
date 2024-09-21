from functions.propagators import CR3BP
from functions.read_ICs import read_ICs
from functions.plot_trajectories import plot_trajectories

# from numpy.random import seed
# seed(0)
from functions.dwell_times import (
    r_rate,
    hist_linear_density,
    kern_linear_density,
    altitude_blocks,
)

# ids = [25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600]
# ids = [1075, 1100, 1125, 1140]
ids = [25, 100, 150, 200, 400, 600]
tf, DRO_ICs = read_ICs("data/ICs_DRO.csv", ids)
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

hist_linear_density(
    CR3BP_states,
    times,
    tf * base_propagator.TU,
    (-base_propagator.mu) * base_propagator.LU,
    ids,
)
