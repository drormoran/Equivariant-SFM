import cv2  # DO NOT REMOVE
from datasets import SceneData, ScenesDataSet
import train
from utils import general_utils, path_utils
from utils.Phases import Phases
import torch


def train_single_model(conf, device, phase):
    # Create data
    scene_data = SceneData.create_scene_data(conf)

    # Create model
    model = general_utils.get_class("models." + conf.get_string("model.type"))(conf).to(device)
    if phase is Phases.FINE_TUNE:
        path = path_utils.path_to_model(conf, Phases.TRAINING)
        model.load_state_dict(torch.load(path))

    # Sequential Optimization
    if conf.get_bool("train.sequential", default=False):
        n_cams = scene_data.y.shape[0]
        conf['train']['num_of_epochs'] = 1000
        conf['train']['scheduler_milestone'] = []
        for subset_size in range(2, n_cams):
            print("########## Train model on subset of size {} ##########".format(subset_size))
            subset_data = SceneData.get_subset(scene_data, subset_size)
            conf["dataset"]["scan"] = subset_data.scan_name
            dubscene_dataset = ScenesDataSet.ScenesDataSet([subset_data], return_all=True)
            subscene_loader = ScenesDataSet.DataLoader(dubscene_dataset).to(device)
            _, _, _, _ = train.train(conf, subscene_loader, model, phase)

        conf['train']['num_of_epochs'] = 20000
        conf['train']['scheduler_milestone'] = [10000]
        conf["dataset"]["scan"] = scene_data.scan_name

    # Optimize Scene
    scene_dataset = ScenesDataSet.ScenesDataSet([scene_data], return_all=True)
    scene_loader = ScenesDataSet.DataLoader(scene_dataset).to(device)
    train_stat, train_errors, _, _ = train.train(conf, scene_loader, model, phase)

    # Write results
    train_errors.drop("Mean", inplace=True)
    train_stat["Scene"] = train_errors.index
    train_stat.set_index("Scene", inplace=True)
    train_res = train_errors.join(train_stat)
    general_utils.write_results(conf, train_res, file_name="Results_" + phase.name, append=True)


if __name__ == "__main__":
    conf, device, phase = general_utils.init_exp(Phases.OPTIMIZATION.name)
    train_single_model(conf, device, phase)
