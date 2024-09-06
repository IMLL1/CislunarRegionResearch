import pandas as pd
import numpy as np
from numpy.typing import ArrayLike


def read_ICs(ids: ArrayLike | int) -> np.ndarray:
    ICs = pd.read_csv("DRO_ICs.csv", index_col=0, usecols=[*range(7), 8])

    ICs = ICs.loc[ids]
    period = np.array(ICs["Period"])
    pv = np.array(ICs[["x0", "y0", "z0", "vx0", "vy0", "vz0"]])

    return period, pv
