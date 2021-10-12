import torch
from datetime import datetime
from scipy.io import savemat
import shutil
from pyhocon import HOCONConverter,ConfigTree
import sys
import json
from pyhocon import ConfigFactory
import argparse
import os
import numpy as np
import pandas as pd
import portalocker
from utils.Phases import Phases
from utils.path_utils import path_to_exp, path_to_cameras, path_to_code_logs, path_to_conf
import random


def log_code(conf):
    code_path = path_to_code_logs(conf)

    files_to_log = ["train.py", "single_scene_optimization.py", "multiple_scenes_learning.py", "loss_functions.py"]
    for file_name in files_to_log:
        shutil.copyfile('{}'.format(file_name), os.path.join(code_path, file_name))

    dirs_to_log = ["datasets", "models"]
    for dir_name in dirs_to_log:
        shutil.copytree('{}'.format(dir_name), os.path.join(code_path, dir_name))

    # Print conf
    with open(os.path.join(code_path, 'exp.conf'), 'w') as conf_log_file:
        conf_log_file.write(HOCONConverter.convert(conf, 'hocon'))


def save_camera_mat(conf, save_cam_dict, scan, phase, epoch=None):
    path_cameras = path_to_cameras(conf, phase, epoch=epoch, scan=scan)
    np.savez(path_cameras, **save_cam_dict)
    savemat(path_cameras, save_cam_dict)


def write_results(conf, df, file_name="Results", append=False):
    exp_path = path_to_exp(conf)
    results_file_path = os.path.join(exp_path, '{}.xlsx'.format(file_name))

    if append:
        locker_file = os.path.join(exp_path, '{}.lock'.format(file_name))
        lock = portalocker.Lock(locker_file, timeout=1000)
        with lock:
            if os.path.exists(results_file_path):
                prev_df = pd.read_excel(results_file_path).set_index("Scene")
                merged_err_df = prev_df.append(df)
            else:
                merged_err_df = df

            merged_err_df.to_excel(results_file_path)
    else:
        df.to_excel(results_file_path)


def init_exp_version():
    return '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_error(err_string):
    print(err_string, file=sys.stderr)


def config_tree_to_string(config):
    config_dict={}
    for it in config.keys():
        if isinstance(config[it],ConfigTree):
            it_dict = {key:val for key,val in config[it].items()}
            config_dict[it]=it_dict
        else:
            config_dict[it] = config[it]
    return json.dumps(config_dict)


def bmvm(bmats, bvecs):
    return torch.bmm(bmats, bvecs.unsqueeze(-1)).squeeze()


def get_full_conf_vals(conf):
    # return a conf file as a dictionary as follow:
    # "key.key.key...key": value
    # Useful for the conf.put() command
    full_vals = {}
    for key, val in conf.items():
        if isinstance(val, dict):
            part_vals = get_full_conf_vals(val)
            for part_key, part_val in part_vals.items():
                full_vals[key + "." +part_key] = part_val
        else:
            full_vals[key] = val

    return full_vals


def parse_external_params(ext_params_str, conf):
    for param in ext_params_str.split(','):
        key_val = param.split(':')
        if len(key_val) == 3:
            conf[key_val[0]][key_val[1]] = key_val[2]
        elif len(key_val) == 2:
            conf[key_val[0]] = key_val[1]
    return conf


def init_exp(default_phase):
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str)
    parser.add_argument('--scan', type=str, default=None)
    parser.add_argument('--exp_version', type=str, default=None)
    parser.add_argument('--external_params', type=str, default=None)
    parser.add_argument('--phase', type=str, default=default_phase)
    opt = parser.parse_args()

    # Init Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Init Conf
    conf_file_path = path_to_conf(opt.conf)
    conf = ConfigFactory.parse_file(conf_file_path)
    conf["original_file_name"] = opt.conf

    # Init external params
    if opt.external_params is not None:
        conf = parse_external_params(opt.external_params, conf)

    # Init Version
    if opt.exp_version is None:
        exp_version = init_exp_version()
    else:
        exp_version = opt.exp_version
    conf['exp_version'] = exp_version

    # Init scan
    if opt.scan is not None:
        conf['dataset']['scan'] = opt.scan
    elif 'scan' not in conf['dataset'].keys():
        conf['dataset']['scan'] = 'Multiple_Scenes'

    # Init Seed
    seed = conf.get_int('random_seed', default=None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Init Phase
    phase = Phases[opt.phase]

    return conf, device, phase
