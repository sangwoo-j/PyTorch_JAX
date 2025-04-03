# ------------------------------- #
# Collocation points
# and Boundary points

import numpy as np
from scipy.stats.qmc import LatinHypercube

def lhs_sampling_coll(N_coll=100, x_range=(0, 1), y_range=(0, 1)):
    lhs = LatinHypercube(d=2, scramble=True)
    samples_coll = lhs.random(n=N_coll)

    x_samples_coll = (x_range[0] + (x_range[1] - x_range[0]) * samples_coll[:, 0]).reshape(-1, 1)
    y_samples_coll = (y_range[0] + (y_range[1] - y_range[0]) * samples_coll[:, 1]).reshape(-1, 1)

    coll_points = np.hstack((x_samples_coll, y_samples_coll))

    return coll_points

def lhs_sampling_bc1(N_bc1=100, y_range=(0, 1)):
    lhs = LatinHypercube(d=1, scramble=True)
    samples_bc1 = lhs.random(n=N_bc1)

    y_bc1 = (y_range[0] + (y_range[1] - y_range[0]) * samples_bc1[:, 0]).reshape(-1, 1)
    x_bc1 = np.full_like(y_bc1, 0)

    bc1_points = np.hstack((x_bc1, y_bc1))

    return bc1_points

def lhs_sampling_bc2(N_bc2=100, x_range=(0, 1)):
    lhs = LatinHypercube(d=1, scramble=True)
    samples_bc2 = lhs.random(n=N_bc2)

    x_bc2 = (x_range[0] + (x_range[1] - x_range[0]) * samples_bc2[:, 0]).reshape(-1, 1)
    y_bc2 = np.full_like(x_bc2, 1)

    bc2_points = np.hstack((x_bc2, y_bc2))

    return bc2_points

def lhs_sampling_bc3(N_bc3=100, y_range=(0, 1)):
    lhs = LatinHypercube(d=1, scramble=True)
    samples_bc3 = lhs.random(n=N_bc3)

    y_bc3 = (y_range[0] + (y_range[1] - y_range[0]) * samples_bc3[:, 0]).reshape(-1, 1)
    x_bc3 = np.full_like(y_bc3, 1)

    bc3_points = np.hstack((x_bc3, y_bc3))

    return bc3_points

def lhs_sampling_bc4(N_bc4=100, x_range=(0, 1)):
    lhs = LatinHypercube(d=1, scramble=True)
    samples_bc4 = lhs.random(n=N_bc4)

    x_bc4 = (x_range[0] + (x_range[1] - x_range[0]) * samples_bc4[:, 0]).reshape(-1, 1)
    y_bc4 = np.full_like(x_bc4, 0)

    bc4_points = np.hstack((x_bc4, y_bc4))

    return bc4_points

N_coll = 1000
N_bc = 100

coll_points = lhs_sampling_coll(N_coll)
bc1_points = lhs_sampling_bc1(N_bc)
bc2_points = lhs_sampling_bc2(N_bc)
bc3_points = lhs_sampling_bc3(N_bc)
bc4_points = lhs_sampling_bc4(N_bc)

np.savez("./data/sampling_points.npz",
          coll_points = coll_points,
          bc1_points = bc1_points,
          bc2_points=bc2_points,
          bc3_points=bc3_points,
          bc4_points=bc4_points,
        )