import cv2  # DO NOT REMOVE
from utils import general_utils, dataset_utils
from utils.Phases import Phases
from datasets.ScenesDataSet import ScenesDataSet, DataLoader
from datasets import SceneData
from single_scene_optimization import train_single_model
import train
import copy


def main():
    # Init Experiment
    conf, device, phase = general_utils.init_exp(Phases.TRAINING.name)
    general_utils.log_code(conf)

    # Get configuration
    min_sample_size = conf.get_int('dataset.min_sample_size')
    max_sample_size = conf.get_int('dataset.max_sample_size')
    batch_size = conf.get_int('dataset.batch_size')
    optimization_num_of_epochs = conf.get_int("train.optimization_num_of_epochs")
    optimization_eval_intervals = conf.get_int('train.optimization_eval_intervals')
    optimization_lr = conf.get_float('train.optimization_lr')

    # Create train, test and validation sets
    test_scenes = SceneData.create_scene_data_from_list(conf.get_list('dataset.test_set'), conf)
    validation_scenes = SceneData.create_scene_data_from_list(conf.get_list('dataset.validation_set'), conf)
    train_scenes = SceneData.create_scene_data_from_list(conf.get_list('dataset.train_set'), conf)

    train_set = ScenesDataSet(train_scenes, return_all=False, min_sample_size=min_sample_size, max_sample_size=max_sample_size)
    validation_set = ScenesDataSet(validation_scenes, return_all=True)
    test_set = ScenesDataSet(test_scenes, return_all=True)

    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True).to(device)
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False).to(device)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False).to(device)

    # Train model
    model = general_utils.get_class("models." + conf.get_string("model.type"))(conf).to(device)
    train_stat, train_errors, validation_errors, test_errors = train.train(conf, train_loader, model, phase, validation_loader, test_loader)

    # Write results
    general_utils.write_results(conf, train_stat, file_name="Train_Stats")
    general_utils.write_results(conf, train_errors, file_name="Train")
    general_utils.write_results(conf, validation_errors, file_name="Validation")
    general_utils.write_results(conf, test_errors, file_name="Test")

    # Send jobs for fine-tuning and short optimization
    test_scans_list = []
    for test_data in test_set:
        test_scans_list.append(test_data.scan_name)

    conf_test = copy.deepcopy(conf)
    conf_test['dataset']['scans_list'] = test_scans_list
    conf_test['train']['num_of_epochs'] = optimization_num_of_epochs
    conf_test['train']['eval_intervals'] = optimization_eval_intervals
    conf_test['train']['lr'] = optimization_lr

    optimization_all_sets(conf_test, device, Phases.FINE_TUNE)
    optimization_all_sets(conf_test, device, Phases.SHORT_OPTIMIZATION)


def optimization_all_sets(conf, device, phase):
    # Get logs directories
    scans_list = conf.get_list('dataset.scans_list')
    for i, scan in enumerate(scans_list):
        conf["dataset"]["scan"] = scan
        train_single_model(conf, device, phase)


if __name__ == "__main__":
    main()
