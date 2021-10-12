import sys
sys.path.append("../bundle_adjustment/ceres-solver/ceres-bin/lib/") # so

import PyCeres
import numpy as np
import scipy.io as sio
import cv2
from utils import geo_utils


def order_cam_param_for_c(Rs, ts, Ks):
    """
    Orders a [m, 12] matrix for the ceres function as follows:
    Ps_for_c[i, 0:3] 3 parameters for the vector representing the rotation
    Ps_for_c[i, 3:6] 3 parameters for the location of the camera
    Ps_for_c[i, 6:11] 5 parameters for the upper triangular part of the calibration matrix
    :param Rs: [m,3,3]
    :param ts: [m,3]
    :param Ks: [m,3,3]
    :return: Ps_for_c [m, 12]
    """
    n_cam = len(Rs)
    Ps_for_c = np.zeros([n_cam, 12])
    for i in range(n_cam):
        Ps_for_c[i, 0:3] = cv2.Rodrigues(Rs[i].T)[0].T
        Ps_for_c[i, 3:6] = (-Rs[i].T @ ts[i].reshape([3, 1])).T
        Ps_for_c[i, 6:11] = [Ks[i, 0, 0], Ks[i, 0, 1], Ks[i, 0, 2], Ks[i, 1, 1], Ks[i, 1, 2]]
        Ps_for_c[i, -1] = 1.0
    return Ps_for_c


def reorder_from_c_to_py(Ps_for_c, Ks):
    """
    Read back the camera parameters from the
    :param Ps_for_c:
    :return: Rs, ts, Ps
    """
    n_cam = len(Ps_for_c)
    Rs = np.zeros([n_cam, 3, 3])
    ts = np.zeros([n_cam, 3])
    Ps = np.zeros([n_cam, 3,4])
    for i in range(n_cam):
        Rs[i] = cv2.Rodrigues(Ps_for_c[i, 0:3])[0].T
        ts[i] = -Rs[i] @ Ps_for_c[i, 3:6].reshape([3, 1]).flatten()
        Ps[i] = geo_utils.get_camera_matrix(R=Rs[i], t=ts[i], K=Ks[i])
    return Rs, ts, Ps


def run_euclidean_ceres(Xs, xs, Rs, ts, Ks, point_indices):
    """
    Calls a c++ function that optimizes the camera parameters and the 3D points for a lower reprojection error.
    :param Xs: [n, 3]
    :param xs: [v,2]
    :param Rs: [m,3,3]
    :param ts: [m,3]
    :param Ks: [m,3,3]
    :param point_indices: [2,v]
    :return:
    new_Rs, new_ts, new_Ps, new_Xs Which have a lower reprojection error
    """
    if Xs.shape[-1] == 4:
        Xs = Xs[:,:3]
    assert Xs.shape[-1] == 3
    assert xs.shape[-1] == 2
    n_cam = len(Rs)
    n_pts = Xs.shape[0]
    n_observe = xs.shape[0]

    Ps_for_c = order_cam_param_for_c(Rs, ts, Ks).astype(np.double)
    Xs_flat = Xs.flatten("C").astype(np.double)
    Ps_for_c_flat = Ps_for_c.flatten("C").astype(np.double)
    xs_flat = xs.flatten("C").astype(np.double)
    point_indices = point_indices.flatten("C")

    Xsu = np.zeros_like(Xs_flat)
    Psu = np.zeros_like(Ps_for_c_flat)

    PyCeres.eucPythonFunctionOursBA(Xs_flat, xs_flat, Ps_for_c_flat, point_indices, Xsu, Psu, n_cam, n_pts, n_observe)

    new_Ps_for_c = Ps_for_c + Psu.reshape([n_cam, 12], order="C")

    new_Rs, new_ts, new_Ps = reorder_from_c_to_py(new_Ps_for_c, Ks)
    new_Xs = Xs + Xsu.reshape([n_pts,3], order="C")

    return new_Rs, new_ts, new_Ps, new_Xs


def run_projective_ceres(Ps, Xs, xs, point_indices):
    """
    Calls the c++ function, that loops over the variables:
    for i in range(v):
        xs[2*i], xs[2*i + 1], Ps + 12 * (camIndex), Xs + 3 * (point3DIndex)
    :param Ps: [m,  3, 4]
    :param Xs: [n, 3]
    :param xs: [v, 2]
    :param point_indices: [2,v]
    :return: new_Ps: [m, 12]
            new_Xs: [n,3]
    """
    if Xs.shape[-1] == 4:
        Xs = Xs[:,:3]
    assert Xs.shape[-1] == 3
    assert xs.shape[-1] == 2
    m = Ps.shape[0]
    n = Xs.shape[0]
    v = point_indices.shape[1]
    Ps_single_flat = Ps.reshape([-1, 12], order="F")   #  [m, 12] Each camera is in *column* major as in matlab! the cpp code assumes it because the original code was in matlab

    Ps_flat = Ps_single_flat.flatten("C")  # row major as in python
    Xs_flat = Xs.flatten("C")
    xs_flat = xs.flatten("C")
    point_idx_flat = point_indices.flatten("C")

    Psu = np.zeros_like(Ps_flat)
    Xsu = np.zeros_like(Xs_flat)

    PyCeres.pythonFunctionOursBA(Xs_flat, xs_flat, Ps_flat, point_idx_flat, Xsu, Psu, m, n, v)
    Psu = Psu.reshape([m,12], order="C")
    Psu = Psu.reshape([m,3,4], order="F")  #  [m, 12] Each camera is in *column* major as in matlab! the cpp code assumes it because the original code was in matlab
    Xsu = Xsu.reshape([n,3])

    new_Ps = Ps + Psu
    new_Xs = Xs + Xsu

    return new_Ps, new_Xs

def run_euclidean_python_ceres(Xs, xs, Rs, ts, Ks, point_indices, print_out=True):
    """
    Calls a c++ function that optimizes the camera parameters and the 3D points for a lower reprojection error.
    :param Xs: [n, 3]
    :param xs: [v,2]
    :param Rs: [m,3,3]
    :param ts: [m,3]
    :param Ks: [m,3,3]
    :param point_indices: [2,v]
    :return:
    new_Rs, new_ts, new_Ps, new_Xs Which have a lower reprojection error
    """
    if Xs.shape[-1] == 4:
        Xs = Xs[:,:3]
    assert Xs.shape[-1] == 3
    assert xs.shape[-1] == 2
    n_cam = len(Rs)
    n_pts = Xs.shape[0]
    n_observe = xs.shape[0]

    Ps_for_c = order_cam_param_for_c(Rs, ts, Ks).astype(np.double)
    Xs_flat = Xs.flatten("C").astype(np.double)
    Ps_for_c_flat = Ps_for_c.flatten("C").astype(np.double)
    xs_flat = xs.flatten("C").astype(np.double)
    point_indices = point_indices.flatten("C")

    Xsu = np.zeros_like(Xs_flat)
    Psu = np.zeros_like(Ps_for_c_flat)

    problem = PyCeres.Problem()
    for i in range(n_observe):  # loop over the observations
        camIndex = int(point_indices[i])
        point3DIndex = int(point_indices[i + n_observe])

        cost_function = PyCeres.eucReprojectionError(xs_flat[2 * i], xs_flat[2 * i + 1],
                                                      Ps_for_c_flat[12 * camIndex:12 * (camIndex + 1)],
                                                      Xs_flat[3 * point3DIndex:3 * (point3DIndex + 1)])

        loss_function = PyCeres.HuberLoss(0.1)
        problem.AddResidualBlock(cost_function, loss_function, Psu[12 * camIndex:12 * (camIndex + 1)],
                                 Xsu[3 * point3DIndex:3 * (point3DIndex + 1)])

    options = PyCeres.SolverOptions()

    options.function_tolerance = 0.0001
    options.max_num_iterations = 100
    options.num_threads = 24

    options.linear_solver_type = PyCeres.LinearSolverType.DENSE_SCHUR
    options.minimizer_progress_to_stdout = True
    if not print_out:
        PyCeres.LoggingType = PyCeres.LoggingType.SILENT

    summary = PyCeres.Summary()
    PyCeres.Solve(options, problem, summary)
    if print_out:
        print(summary.FullReport())

    if ~Psu.any():
        print('Warning no change to Ps')
    if ~Xsu.any():
        print('Warning no change to Xs')

    new_Ps_for_c = Ps_for_c + Psu.reshape([n_cam, 12], order="C")

    new_Rs, new_ts, new_Ps = reorder_from_c_to_py(new_Ps_for_c, Ks)
    new_Xs = Xs + Xsu.reshape([n_pts,3], order="C")

    return new_Rs, new_ts, new_Ps, new_Xs


def run_projective_python_ceres(Ps, Xs, xs, point_indices, print_out=True):
    """
    Calls the c++ function, that loops over the variables:
    for i in range(v):
        xs[2*i], xs[2*i + 1], Ps + 12 * (camIndex), Xs + 3 * (point3DIndex)
    :param Ps: [m,  3, 4]
    :param Xs: [n, 3]
    :param xs: [v, 2]
    :param point_indices: [2,v]
    :return: new_Ps: [m, 12]
            new_Xs: [n,3]
    """
    if Xs.shape[-1] == 4:
        Xs = Xs[:,:3]
    assert Xs.shape[-1] == 3
    assert xs.shape[-1] == 2
    m = Ps.shape[0]
    n = Xs.shape[0]
    v = point_indices.shape[1]
    Ps_single_flat = Ps.reshape([-1, 12], order="F")   #  [m, 12] Each camera is in *column* major as in matlab! the cpp code assumes it because the original code was in matlab

    Ps_flat = Ps_single_flat.flatten("C").astype(np.double)  # row major as in python
    Xs_flat = Xs.flatten("C").astype(np.double)
    xs_flat = xs.flatten("C")
    point_idx_flat = point_indices.flatten("C")

    Psu = np.zeros_like(Ps_flat)
    Xsu = np.zeros_like(Xs_flat)

    problem = PyCeres.Problem()
    for i in range(v):  # loop over the observations
        camIndex = int(point_idx_flat[i])
        point3DIndex = int(point_idx_flat[i + v])

        cost_function = PyCeres.projReprojectionError(xs_flat[2*i], xs_flat[2*i + 1], Ps_flat[12*camIndex:12*(camIndex+1)], Xs_flat[3 *point3DIndex:3*(point3DIndex+1)])

        loss_function = PyCeres.HuberLoss(0.1)
        problem.AddResidualBlock(cost_function, loss_function, Psu[12*camIndex:12*(camIndex+1)], Xsu[3 *point3DIndex:3*(point3DIndex+1)])


    options = PyCeres.SolverOptions()

    options.function_tolerance = 0.0001
    options.max_num_iterations = 100
    options.num_threads = 24

    options.linear_solver_type = PyCeres.LinearSolverType.DENSE_SCHUR
    options.minimizer_progress_to_stdout = True

    summary = PyCeres.Summary()
    PyCeres.Solve(options, problem, summary)
    if print_out:
        print(summary.FullReport())
    Psu = Psu.reshape([m,12], order="C")
    Psu = Psu.reshape([m,3,4], order="F")  #  [m, 12] Each camera is in *column* major as in matlab! the cpp code assumes it because the original code was in matlab
    Xsu = Xsu.reshape([n,3])

    new_Ps = Ps + Psu
    new_Xs = Xs + Xsu

    return new_Ps, new_Xs