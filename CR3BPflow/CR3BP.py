from scipy.optimize import newton, minimize, approx_fprime
from scipy.integrate import solve_ivp
import pandas as pd
import numpy as np
import numdifftools as nd


class CR3BP:
    def __init__(self, mu: float = 1.215058560962404e-2, odemethod: str = "DOP853"):
        self.solver = odemethod
        if mu < 0.5:
            self.mu = mu
        else:
            print("mu should be <0.5. Setting mu=1-mu")
            self.mu = 1 - mu

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

        lpoints = np.array([L1, L2, L3, L4, L5]).T

    def U(self, pos):
        x, y, z = pos
        r1mag = np.sqrt((x + self.mu) ** 2 + y**2 + z**2)
        r2mag = np.sqrt((x - 1 + self.mu) ** 2 + y**2 + z**2)
        Ugrav = -((1 - self.mu) / r1mag + self.mu / r2mag)
        Ucent = -0.5 * (x**2 + y**2)

        return Ugrav + Ucent

    def get_JC(self, x, y, z, dx, dy, dz):
        JC = -2 * self.U([x, y, z])
        JC -= dx**2 + dy**2 + dz**2
        return JC

    def eom(self, t, state):
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
        tfin,
        n_eval=1000,
        tol=1e-14,
        # dense_output=False,
    ):

        t_eval = np.linspace(0, tfin, n_eval)

        soln = solve_ivp(
            fun=self.eom,
            t_span=(0, tfin),
            t_eval=t_eval,
            y0=np.array(state0),
            method=self.solver,
            atol=tol,
            rtol=100 * tol,
            # dense_output=dense_output,
        )

        return soln.y.T
        # self.t = self.solution.t

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

            states = self.propagate(states_in[1:], states_in[0], n_eval=2, tol=tol)
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

    def A(self, state):
        # ddU = nd.Hessian(self.U)
        A11 = np.zeros((3, 3), dtype=float)
        A12 = np.identity(3)
        # A21 = ddU(state)
        A22 = np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 0]])

        f = lambda x: self.eom(0, x)
        A21 = approx_fprime(state, f)[3:, :3]
        A = np.block([[A11, A12], [A21, A22]])
        return A

    def coupled_stm_eom(self, t, state):
        pv = state[:6]
        dpv = self.eom(t, pv)
        stm = state[6:].reshape((6, 6))
        A = self.A(pv)  # pv[:3]
        dstm = A @ stm

        dstate = np.array([*dpv, *dstm.flatten()])
        return dstate

    def get_stm(
        self,
        x0,
        t0,
        tf,
        tol=1e-14,
    ):

        soln = solve_ivp(
            fun=self.coupled_stm_eom,
            t_span=(t0, tf),
            t_eval=[tf],
            y0=np.array([*x0, *np.eye(6).flatten()]),
            method=self.solver,
            atol=tol,
            rtol=100 * tol,
        )

        return soln.y[6:].reshape(6, 6)

    def monodromy(self, x0, period, tol=1e-13):
        Phi = self.get_stm(x0, 0, period / 2, tol)
        G = np.diag([1, -1, 1, -1, 1, -1])
        Omega = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        I = np.identity(3)
        O = np.zeros((3, 3))
        mtx1 = np.block([[O, -I], [I, -2 * Omega]])
        mtx2 = np.block([[-2 * Omega, I], [-I, O]])
        M = G @ mtx1 @ Phi.T @ mtx2 @ G @ Phi
        return M

    def manifold_points(self, x0, period, npts=25, d=1e-4, tol=3e-16):
        tfs = np.linspace(0, period, npts + 1)[1:]
        M = self.monodromy(x0, period, tol)
        evals, evecs = np.linalg.eig(M)
        order = np.argsort(np.abs(evals))
        vs = evecs[:, order[[0, 1, -1, -2]]]
        assert bool(np.all(np.isreal(vs)))
        vs = np.real(vs)

        soln = solve_ivp(
            fun=self.coupled_stm_eom,
            t_span=(0, period),
            t_eval=tfs,
            y0=np.array([*x0, *np.eye(6).flatten()]),
            method=self.solver,
            atol=tol,
            rtol=100 * tol,
        ).y

        xs = [soln[:6, i] for i in range(npts)]
        stms = [soln[6:, i].reshape(6, 6) for i in range(npts)]

        x0s_s1 = []
        x0s_s2 = []
        x0s_u1 = []
        x0s_u2 = []
        for i in range(npts):
            x = xs[i]
            stm = stms[i]
            vs_i = stm @ vs
            posnorms = np.linalg.norm(vs_i[:3, :], axis=0)
            vs_i = vs_i / posnorms

            # make sure all vecs are real only
            x0s_s1.append(x + d * vs_i[:, 0])
            x0s_s2.append(x + d * vs_i[:, 1])
            x0s_u1.append(x + d * vs_i[:, -1])
            x0s_u2.append(x + d * vs_i[:, -2])

        return x0s_s1, x0s_s2, x0s_u1, x0s_u2

    def manifold_curves(
        self, x0, period, npts=25, d=1e-4, nprop: int = 1000, tprop=None, tol=1e-10
    ):
        if tprop == None:
            tprop = period / 3

        xs_s1 = []
        xs_s2 = []
        xs_u1 = []
        xs_u2 = []
        x0s_s1, x0s_s2, x0s_u1, x0s_u2 = self.manifold_points(x0, period, npts, d, tol)
        for i in range(npts):
            xs_s1.append(self.propagate(x0s_s1[i], -tprop, nprop, tol))
            xs_s2.append(self.propagate(x0s_s2[i], -tprop, nprop, tol))
            xs_u1.append(self.propagate(x0s_u1[i], tprop, nprop, tol))
            xs_u2.append(self.propagate(x0s_u2[i], tprop, nprop, tol))

        return xs_s1, xs_s2, xs_u1, xs_u2
