import scipy.io as sio
import numpy as np
import os


def read_mat_files(path):
    raw_data = sio.loadmat(path + '.mat', squeeze_me=True)
    Xs = raw_data['Points3D'].T
    M = raw_data['M']
    m, n = M.shape
    m = m // 2
    xs = M.reshape([m, 2, n]).transpose([0, 2, 1])
    Ps = np.stack(raw_data['Ps'])
    data = {'Ps': Ps, 'Xs': Xs, 'xs': xs}
    return data

def read_euc_gt_mat_files(path):
    raw_data = sio.loadmat(path + '.mat', squeeze_me=True)
    M = raw_data['M']
    if not isinstance(M, (np.ndarray, np.generic) ):
        M = np.asarray(M.todense())
    Rs = np.stack(raw_data['R_gt'])
    ts = np.stack(raw_data['T_gt'])
    Ks = np.stack(raw_data['K_gt'])
    m, n = M.shape
    m = m // 2
    xs = M.reshape([m, 2, n]).transpose([0, 2, 1])
    data = {'Rs': Rs, 'ts': ts, 'Ks':Ks, 'xs': xs}
    return data

def read_proj_gt_mat_files(path):
    raw_data = sio.loadmat(path + '.mat', squeeze_me=True)
    M = np.asarray(raw_data['M'])
    m, n = M.shape
    m = m // 2
    xs = M.reshape([m, 2, n]).transpose([0, 2, 1])
    data = {'xs': xs}
    return data

def read_euc_our_mat_files(path, name='Final_Cameras'):
    raw_data = sio.loadmat(os.path.join(path, 'cameras', name) + '.mat', squeeze_me=True)
    Xs = raw_data['pts3D'][:3].T.astype(np.double)
    Rs = raw_data['Rs']
    ts = raw_data['ts']
    Ks = raw_data['Ks']
    data = {'Xs': Xs, 'Rs': Rs, 'ts': ts, 'Ks':Ks}
    return data


def read_proj_our_mat_files(path, name='Final_Cameras'):
    raw_data = sio.loadmat(os.path.join(path, 'cameras', name) + '.mat', squeeze_me=True)
    Xs = raw_data['pts3D'][:3].T.astype(np.double)
    Ps = raw_data['Ps']
    data = {'Ps': Ps, 'Xs':Xs}
    return data

