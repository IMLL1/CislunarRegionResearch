import pandas as pd
import numpy as np
from numpy.typing import ArrayLike


def read_ICs(ic_file: str, ids: ArrayLike | int) -> np.ndarray:
    ICs = pd.read_csv(ic_file, index_col=0, usecols=[*range(7), 8])

    ICs = ICs.loc[ids]
    period = np.array(ICs["Period"])
    pv = np.array(ICs[["x0", "y0", "z0", "vx0", "vy0", "vz0"]])

    return period, pv
