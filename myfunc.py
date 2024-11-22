import numpy as np
from numba import njit


@njit(cache=True)
def Min(x, y):
    x = np.asarray(x).reshape((-1,))
    y = np.asarray(y).reshape((-1,))

    z = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] <= y[i]:
            z[i] = x[i]
        else:
            z[i] = y[i]
    return z


@njit(cache=True)
def dMindx(x, y):
    x = np.asarray(x).reshape((-1,))
    y = np.asarray(y).reshape((-1,))

    z = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] <= y[i]:
            z[i] = 1
        else:
            z[i] = 0
    return z


@njit(cache=True)
def dMindy(x, y):
    return 1-dMindx(x, y)
