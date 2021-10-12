import cv2  # Do not remove
import torch
from utils import geo_utils, general_utils, dataset_utils, path_utils
import scipy.io as sio
import numpy as np
import os.path


def get_raw_data(conf, scan):
    """
    :param conf:
    :return:
    M - Points Matrix (2mxn)
    Ns - Normalization matrices (mx3x3)
    Ps_gt - Olsson's estimated camera matrices (mx3x4)
    NBs - Normzlize Bifocal Tensor (Normalized Fn) (3mx3m)
    triplets
    """
    # Init
    dataset_path_format = os.path.join(path_utils.path_to_datasets(), 'Projective', '{}.npz')

    # Get conf parameters
    if scan is None:
        scan = conf.get_string('dataset.scan')
    use_gt = conf.get_bool('dataset.use_gt')

    # Get raw data
    dataset = np.load(dataset_path_format.format(scan))

    # Get bifocal tensors and 2D points
    M = dataset['M']
    Ps_gt = dataset['Ps_gt']
    Ns = dataset['Ns']

    if use_gt:
        M = torch.from_numpy(dataset_utils.correct_matches_global(M, Ps_gt, Ns))

    M = torch.from_numpy(M).float()
    Ps_gt = torch.from_numpy(Ps_gt).float()
    Ns = torch.from_numpy(Ns).float()

    return M, Ns, Ps_gt


def test_Ps_M(Ps, M, Ns):
    global_rep_err = geo_utils.calc_global_reprojection_error(Ps.numpy(), M.numpy(), Ns.numpy())
    print("Reprojection Error: Mean = {}, Max = {}".format(np.nanmean(global_rep_err), np.nanmax(global_rep_err)))


def test_projective_dataset(scan):
    dataset_path_format = os.path.join(path_utils.path_to_datasets(), 'Projective', '{}.npz')

    # Get raw data
    dataset = np.load(dataset_path_format.format(scan))

    # Get bifocal tensors and 2D points
    M = dataset['M']
    Ps_gt = dataset['Ps_gt']
    Ns = dataset['Ns']

    M_gt = torch.from_numpy(dataset_utils.correct_matches_global(M, Ps_gt, Ns)).float()

    M = torch.from_numpy(M).float()
    Ps_gt = torch.from_numpy(Ps_gt).float()
    Ns = torch.from_numpy(Ns).float()

    print("Test Ps and M")
    test_Ps_M(Ps_gt, M, Ns)

    print("Test Ps and M_gt")
    test_Ps_M(Ps_gt, M_gt, Ns)


if __name__ == "__main__":
    scan = "Alcatraz Courtyard"
    test_projective_dataset(scan)


