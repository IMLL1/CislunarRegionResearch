"""
Adapted from Alfonso Gonzalez's class at https://github.com/alfonsogonzalez/AWP
Some inspiration taken from NASA Johnson Space Center's Copernicus: https://www.nasa.gov/general/copernicus/
"""

from scipy.integrate import solve_ivp
from scipy.optimize import newton, minimize
from scipy.spatial.transform import rotation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CR3BP:
    def __init__(
        self, mu: float = 1.215058560962404e-2, LU: float = 1, TU: float = 382981
    ):
        if mu < 0.5:
            self.mu = mu
        else:
            print("mu should be <0.5. Setting mu=1-mu")
            self.mu = 1 - mu

        self.L_points = self.lagranges()
        G = 6.674e-11
        # period should be sqrt(LU^3 / (G * MU))
        self.LU = LU
        self.TU = TU

    def lagranges(self):
        def optFunc(x):
            zero = (
                -(x + self.mu) * (1 - self.mu) / (np.abs(x + self.mu) ** 3)
                - (x - 1 + self.mu) * self.mu / (np.abs(x - 1 + self.mu) ** 3)
                + x
            )
            return zero

        L1 = [newton(optFunc, (1 - self.mu) / 2), 0]
        L2 = [newton(optFunc, (2 - self.mu) / 2), 0]
        L3 = [newton(optFunc, -1), 0]
        L4 = [1 / 2 - self.mu, np.sqrt(3) / 2]
        L5 = [1 / 2 - self.mu, -np.sqrt(3) / 2]

        return np.array([L1, L2, L3, L4, L5]).T

    def pseudopotential(self, x, y, z):
        r1mag = np.sqrt((x + self.mu) ** 2 + y**2 + z**2)
        r2mag = np.sqrt((x - 1 + self.mu) ** 2 + y**2 + z**2)
        Ugrav = -((1 - self.mu) / r1mag + self.mu / r2mag)
        Ucent = -0.5 * (x**2 + y**2)

        return Ugrav + Ucent

    def eom_nondim(self, t, state):
        x, y, z, vx, vy, vz = state
        xyz = state[:3]
        r1vec = xyz + np.array([self.mu, 0, 0])
        r2vec = xyz + np.array([self.mu - 1, 0, 0])
        r1mag = np.linalg.norm(r1vec)
        r2mag = np.linalg.norm(r2vec)

        ddxyz = (
            -(1 - self.mu) * r1vec / r1mag**3
            - self.mu * r2vec / r2mag**3
            + np.array([2 * vy + x, -2 * vx + y, 0])
        )

        dstate = np.zeros(6)
        dstate[:3] = state[3:]
        dstate[3:] = ddxyz
        return dstate

    def propagate(
        self,
        state0,
        tspan,
        propagator="LSODA",
        rtol=1e-9,
        atol=1e-9,
        dense_output=False,
    ):
        self.solution = solve_ivp(
            fun=self.eom_nondim,
            t_span=(0, tspan),
            y0=np.array(state0),
            method=propagator,
            atol=atol,
            rtol=rtol,
            dense_output=dense_output,
        )

        self.states_nd_cr3bp = self.solution.y.T
        self.ts_nd = self.solution.t
        self.states_cr3bp = self.states_nd_cr3bp * self.LU
        self.states_cr3bp[:, 3:] /= self.TU
        self.ts = self.ts_nd * self.TU

    def get_inertial_states(self, t0: float = 0, inc_rad: float = np.deg2rad(5.14)):
        omega = 1 / self.TU
        theta = omega * (self.ts + t0)
        rots = rotation.Rotation.from_euler(
            "XZY", np.array([inc_rad + 0 * theta, theta, 0 * theta]).T
        )
        r_xy = np.linalg.vector_norm(self.states_cr3bp[:, :2], axis=1)
        rot_vel = omega * r_xy * np.vstack((-np.sin(theta), np.cos(theta), 0 * self.ts))
        pos_new = rots.apply(self.states_cr3bp[:, :3])
        vel_new = self.states_cr3bp[:, 3:] + rot_vel.T
        self.states = np.hstack((pos_new, vel_new))
        moonstate_cr3bp = [(1 - self.mu) * self.LU, 0, 0]
        self.moonstate = rots.apply(moonstate_cr3bp)

    def find_periodic_orbit(
        self,
        opt_vars=["tf", "z", "vy"],
        obj_zero=["vx", "y"],
        init_guess=[2.77, 0.82285, 0, 0.05, 0, 0.17, 0],
        tol=None,
    ):
        func_inputs = pd.Series(
            {"tf": 0, "x": 1, "y": 2, "z": 3, "vx": 4, "vy": 5, "vz": 6}
        )
        fixed_vars = list(func_inputs.drop(index=opt_vars).keys())
        init_guess = np.array(init_guess)
        opt_paramnums = list(func_inputs[opt_vars].values)

        def minFunc(inputs):
            states_in = np.zeros(7)

            # insert non-optimization variables
            states_in[func_inputs[fixed_vars]] = init_guess[func_inputs[fixed_vars]]
            states_in[func_inputs[opt_vars]] = inputs
            # prevent time from going to zero (bad optimization)
            if states_in[0] < 0.5 * init_guess[0]:
                states_in[0] = 0.5 * init_guess[0]

            _, states = self.propagate_nondim(states_in[1:], states_in[0])
            state_fin = states[-1, :]

            # get objective states and their norm
            obj_states = np.array(state_fin[func_inputs[obj_zero] - 1])
            obj_func = np.linalg.norm(obj_states)
            return obj_func

        # init input is init guess for non-set variables
        init_input = init_guess[opt_paramnums]
        min_object = minimize(minFunc, init_input, method="Nelder-Mead", tol=tol)
        minimizing_guess = min_object.x
        optimal_state = np.zeros(7)
        optimal_state[func_inputs[fixed_vars]] = init_guess[func_inputs[fixed_vars]]
        optimal_state[func_inputs[opt_vars]] = minimizing_guess
        return optimal_state
