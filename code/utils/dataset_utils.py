import torch
from utils import geo_utils, general_utils, sparse_utils, plot_utils
import numpy as np


def is_valid_sample(data, min_pts_per_cam=10):
    return data.x.pts_per_cam.min().item() >= min_pts_per_cam


def divide_indices_to_train_test(N, n_val, n_test=0):
    perm = np.random.permutation(N)
    test_indices = perm[:n_test] if n_test>0 else []
    val_indices = perm[n_test:n_test+n_val]
    train_indices = perm[n_test+n_val:]
    return train_indices, val_indices, test_indices


def sample_indices(N, num_samples, adjacent):
    if num_samples == 1:  # Return all the data
        indices = np.arange(N)
    else:
        if num_samples < 1:
            num_samples = int(np.ceil(num_samples * N))
        num_samples = max(2, num_samples)
        if num_samples>=N:
            return np.arange(N)
        if adjacent:
            start_ind = np.random.randint(0,N-num_samples+1)
            end_ind = start_ind+num_samples
            indices = np.arange(start_ind, end_ind)
        else:
            indices = np.random.choice(N,num_samples,replace=False)
    return indices


def save_cameras(outputs, conf, curr_epoch, phase):
    xs = outputs['xs']
    M = geo_utils.xs_to_M(xs)
    general_utils.save_camera_mat(conf, outputs, outputs['scan_name'], phase, curr_epoch)


def get_data_statistics(all_data):
    valid_pts = all_data.valid_pts
    valid_pts_stat = valid_pts.sum(dim=0).float()
    stats = {"Max_2d_pt": all_data.M.max().item(), "Num_2d_pts": valid_pts.sum().item(), "n_pts":all_data.M.shape[-1],
             "Cameras_per_pts_mean": valid_pts_stat.mean().item(), "Cameras_per_pts_std": valid_pts_stat.std().item(),
             "Num of cameras": all_data.y.shape[0]}
    return stats


def correct_matches_global(M, Ps, Ns):
    M_invalid_pts = np.logical_not(get_M_valid_points(M))

    Xs = geo_utils.n_view_triangulation(Ps, M, Ns)
    xs = geo_utils.batch_pflat((Ps @ Xs))[:, 0:2, :]

    # Remove invalid points
    xs[np.isnan(xs)] = 0
    xs[np.stack((M_invalid_pts, M_invalid_pts), axis=1)] = 0

    return xs.reshape(M.shape)


def get_M_valid_points(M):
    n_pts = M.shape[-1]

    if type(M) is torch.Tensor:
        M_valid_pts = torch.abs(M.reshape(-1, 2, n_pts)).sum(dim=1) != 0
        M_valid_pts[:, M_valid_pts.sum(dim=0) < 2] = False
    else:
        M_valid_pts = np.abs(M.reshape(-1, 2, n_pts)).sum(axis=1) != 0
        M_valid_pts[:, M_valid_pts.sum(axis=0) < 2] = False

    return M_valid_pts


def M2sparse(M, normalize=False, Ns=None):
    n_pts = M.shape[1]
    n_cams = int(M.shape[0] / 2)

    # Get indices
    valid_pts = get_M_valid_points(M)
    cam_per_pts = valid_pts.sum(dim=0).unsqueeze(1)
    pts_per_cam = valid_pts.sum(dim=1).unsqueeze(1)
    mat_indices = torch.nonzero(valid_pts).T

    # Get Values
    # reshaped_M = M.reshape(n_cams, 2, n_pts).transpose(1, 2)  # [2m, n] -> [m, 2, n] -> [m, n, 2]
    if normalize:
        norm_M = geo_utils.normalize_M(M, Ns)
        mat_vals = norm_M[mat_indices[0], mat_indices[1], :]
    else:
        mat_vals = M.reshape(n_cams, 2, n_pts).transpose(1, 2)[mat_indices[0], mat_indices[1], :]

    mat_shape = (n_cams, n_pts, 2)
    return sparse_utils.SparseMat(mat_vals, mat_indices, cam_per_pts, pts_per_cam, mat_shape)
