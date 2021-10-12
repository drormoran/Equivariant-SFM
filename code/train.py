import time
import torch
import math

import loss_functions
import evaluation
import copy
from utils import path_utils, dataset_utils, plot_utils
from time import time
import pandas as pd
from utils.Phases import Phases


def epoch_train(train_data, model, loss_func, optimizer, scheduler, epoch):
    model.train()
    train_losses = []
    for train_batch in train_data:  # Loop over all sets - 30
        batch_loss = torch.tensor([0.0], device=train_data.device)
        optimizer.zero_grad()
        for curr_data in train_batch:
            if not dataset_utils.is_valid_sample(curr_data):
                print('{} {} has a camera with not enough points'.format(epoch, curr_data.scan_name))
                continue
            pred_cam = model(curr_data)
            loss = loss_func(pred_cam, curr_data)
            batch_loss += loss
            train_losses.append(loss.item())
        if batch_loss.item()>0:
            batch_loss.backward()
            optimizer.step()
    scheduler.step()

    mean_loss = torch.tensor(train_losses).mean()
    return mean_loss, train_losses


def epoch_evaluation(data_loader, model, conf, epoch, phase, save_predictions=False, bundle_adjustment=True):
    errors_list = []
    model.eval()
    with torch.no_grad():
        for batch_data in data_loader:
            for curr_data in batch_data:
                # Get predictions
                begin_time = time()
                pred_cam = model(curr_data)
                pred_time = time() - begin_time

                # Eval results
                outputs = evaluation.prepare_predictions(curr_data, pred_cam, conf, bundle_adjustment)
                errors = evaluation.compute_errors(outputs, conf, bundle_adjustment)

                errors['Inference time'] = pred_time
                errors['Scene'] = curr_data.scan_name

                # Get scene statistics on final evaluation
                if epoch is None:
                    stats = dataset_utils.get_data_statistics(curr_data)
                    errors.update(stats)

                errors_list.append(errors)

                if save_predictions:
                    dataset_utils.save_cameras(outputs, conf, curr_epoch=epoch, phase=phase)
                    if conf.get_bool('dataset.calibrated'):
                        path = plot_utils.plot_cameras_before_and_after_ba(outputs, errors, conf, phase, scan=curr_data.scan_name, epoch=epoch, bundle_adjustment=bundle_adjustment)

    df_errors = pd.DataFrame(errors_list)
    mean_errors = df_errors.mean()
    df_errors = df_errors.append(mean_errors, ignore_index=True)
    df_errors.at[df_errors.last_valid_index(), "Scene"] = "Mean"
    df_errors.set_index("Scene", inplace=True)
    df_errors = df_errors.round(3)
    print(df_errors.to_string(), flush=True)
    model.train()

    return df_errors


def train(conf, train_data, model, phase, validation_data=None, test_data=None):
    num_of_epochs = conf.get_int('train.num_of_epochs')
    eval_intervals = conf.get_int('train.eval_intervals', default=500)
    validation_metric = conf.get_list('train.validation_metric', default=["our_repro"])

    # Loss functions
    loss_func = getattr(loss_functions, conf.get_string('loss.func'))(conf)

    # Optimizer params
    lr = conf.get_float('train.lr')
    scheduler_milestone = conf.get_list('train.scheduler_milestone')
    gamma = conf.get_float('train.gamma', default=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestone, gamma=gamma)

    best_validation_metric = math.inf
    best_epoch = 0
    best_model = torch.empty(0)
    converge_time = -1
    begin_time = time()

    no_ba_during_training = not conf.get_bool('ba.only_last_eval')

    for epoch in range(num_of_epochs):
        mean_train_loss, train_losses = epoch_train(train_data, model, loss_func, optimizer, scheduler, epoch)
        if epoch % 100 == 0:
            print('{} Train Loss: {}'.format(epoch, mean_train_loss))
        if epoch % eval_intervals == 0 or epoch == num_of_epochs - 1:  # Eval current results
            if phase is Phases.TRAINING:
                validation_errors = epoch_evaluation(validation_data, model, conf, epoch, Phases.VALIDATION, save_predictions=True,bundle_adjustment=no_ba_during_training)
            else:
                validation_errors = epoch_evaluation(train_data, model, conf, epoch, phase, save_predictions=True,bundle_adjustment=no_ba_during_training)

            metric = validation_errors.loc[["Mean"], validation_metric].sum(axis=1).values.item()

            if metric < best_validation_metric:
                converge_time = time()-begin_time
                best_validation_metric = metric
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                print('Updated best validation metric: {} time so far: {}'.format(best_validation_metric, converge_time))
                path = path_utils.path_to_model(conf, phase, epoch=epoch)
                torch.save(best_model.state_dict(), path)

    # Eval final model
    train_stat = {}
    print("Evaluate training set")
    run_ba = conf.get_bool('ba.run_ba', default=True)
    train_errors = epoch_evaluation(train_data, best_model, conf, None, phase, save_predictions=True,bundle_adjustment=run_ba)

    if phase is Phases.TRAINING:
        print("Evaluate validation set")
        validation_errors = epoch_evaluation(validation_data, best_model, conf, None, Phases.VALIDATION, save_predictions=True,bundle_adjustment=run_ba)
        print("Evaluate test set")
        test_errors = epoch_evaluation(test_data, best_model, conf, None, Phases.TEST, save_predictions=True,bundle_adjustment=run_ba)
    else:
        validation_errors = None
        test_errors = None

    # Saving the best model
    path = path_utils.path_to_model(conf, phase, epoch=None)
    torch.save(best_model.state_dict(), path)

    train_stat['Convergence time'] = converge_time
    train_stat['best_epoch'] = best_epoch
    train_stat['best_validation_metric'] = best_validation_metric
    train_stat = pd.DataFrame([train_stat])

    return train_stat, train_errors, validation_errors, test_errors
