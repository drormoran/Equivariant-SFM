import os
from utils.Phases import Phases


def join_and_create(path, folder):
    full_path = os.path.join(path, folder)
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    return full_path


def path_to_datasets():
    return os.path.join('..', 'datasets')


def path_to_condition(conf):
    experiments_folder = os.path.join('..', 'results')
    if not os.path.exists(experiments_folder):
        os.mkdir(experiments_folder)
    exp_name = conf.get_string('exp_name')
    return join_and_create(experiments_folder, exp_name)


def path_to_exp(conf):
    exp_ver = conf.get_string('exp_version')
    exp_ver_path = join_and_create(path_to_condition(conf), exp_ver)

    return exp_ver_path


def path_to_phase(conf, phase):
    exp_path = path_to_exp(conf)
    return join_and_create(exp_path, phase.name)


def path_to_scan(conf, phase, scan=None):
    exp_path = path_to_phase(conf, phase)
    scan = conf.get_string("dataset.scan") if scan is None else scan
    return join_and_create(exp_path, scan)


def path_to_model(conf, phase, epoch=None, scan=None):
    if phase in [Phases.TRAINING, Phases.VALIDATION, Phases.TEST]:
        parent_folder = path_to_exp(conf)
    else:
        parent_folder = path_to_scan(conf, phase, scan=scan)

    models_path = join_and_create(parent_folder, 'models')

    if epoch is None:
        model_file_name = "Final_Model.pt"
    else:
        model_file_name = "Model_Ep{}.pt".format(epoch)

    return os.path.join(models_path, model_file_name)


def path_to_learning_data(conf, phase):
    return join_and_create(path_to_condition(conf), phase)


def path_to_cameras(conf, phase, epoch=None, scan=None):
    scan_path = path_to_scan(conf, phase, scan=scan)
    cameras_path = join_and_create(scan_path, 'cameras')

    if epoch is None:
        cameras_file_name = "Final_Cameras"
    else:
        cameras_file_name = "Cameras_Ep{}".format(epoch)

    return os.path.join(cameras_path, cameras_file_name)


def path_to_plots(conf, phase, epoch=None, scan=None):
    scan_path = path_to_scan(conf, phase, scan=scan)
    plots_path = join_and_create(scan_path, 'plots')

    if epoch is None:
        plots_file_name = "Final_plots.html"
    else:
        plots_file_name = "Plot_Ep{}.html".format(epoch)

    return os.path.join(plots_path, plots_file_name)


def path_to_logs(conf, phase):
    phase_path = path_to_phase(conf, phase)
    logs_path = join_and_create(phase_path, "logs")
    return logs_path


def path_to_code_logs(conf):
    exp_path = path_to_exp(conf)
    code_path = join_and_create(exp_path, "code")
    return code_path


def path_to_conf(conf_file):
    return os.path.join( 'confs', conf_file)