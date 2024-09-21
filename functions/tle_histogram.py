import numpy as np
import sklearn.mixture

mu = 3.986004e5


def getSMA(meanmotion: float) -> float:
    # n^2 = mu / a^3
    a = (mu / meanmotion**2) ** (1 / 3)
    return a


def getSMAlist(tlefile):
    tle_file = open(tlefile)
    SMAs = []
    for line in tle_file.readlines():
        if line[0] != "2":
            continue
        meanmotion = float(line[52:63])  # rev/day
        meanmotion *= 2 * np.pi  # rad/day
        meanmotion /= (60 * 60 * 24)  # rad/s
        SMAs.append(getSMA(meanmotion))
    return np.array(SMAs)


def SMA_GMM(tlefile, num_components):
    a = getSMAlist(tlefile).reshape(-1,1)
    GMM = sklearn.mixture.GaussianMixture(num_components, random_state=0).fit(a)

    return GMM