import torch
import os
from utils import geo_utils
import numpy as np

BA_ERROR = None
try:
    from utils import ba_functions
except ModuleNotFoundError as e:
    BA_ERROR = e


def prepare_predictions(data, pred_cam, conf, bundle_adjustment):
    # Take the inputs from pred cam and turn to ndarray
    outputs = {}
    outputs['scan_name'] = data.scan_name
    calibrated = conf.get_bool('dataset.calibrated')

    Ns = data.Ns.cpu().numpy()
    Ns_inv = data.Ns_invT.transpose(1, 2).cpu().numpy()  # Ks for calibrated, a normalization matrix for uncalibrated
    M = data.M.cpu().numpy()
    xs = geo_utils.M_to_xs(M)

    Ps_norm = pred_cam["Ps_norm"].cpu().numpy()  # Normalized camera!!
    Ps = Ns_inv @ Ps_norm  # unnormalized cameras
    pts3D_pred = geo_utils.pflat(pred_cam["pts3D"]).cpu().numpy()

    pts3D_triangulated = geo_utils.n_view_triangulation(Ps, M=M, Ns=Ns)

    outputs['xs'] = xs  # to compute reprojection error later
    outputs['Ps'] = Ps
    outputs['Ps_norm'] = Ps_norm
    outputs['pts3D_pred'] = pts3D_pred  # 4,m
    outputs['pts3D_triangulated'] = pts3D_triangulated  # 4,n

    if calibrated:
        Ks = Ns_inv  # data.Ns.inverse().cpu().numpy()
        outputs['Ks'] = Ks
        Rs_gt, ts_gt = geo_utils.decompose_camera_matrix(data.y.cpu().numpy(), Ks)  # For alignment and R,t errors
        outputs['Rs_gt'] = Rs_gt
        outputs['ts_gt'] = ts_gt

        Rs_pred, ts_pred = geo_utils.decompose_camera_matrix(Ps_norm)
        outputs['Rs'] = Rs_pred
        outputs['ts'] = ts_pred

        Rs_fixed, ts_fixed, similarity_mat = geo_utils.align_cameras(Rs_pred, Rs_gt, ts_pred, ts_gt, return_alignment=True) # Align  Rs_fixed, tx_fixed
        outputs['Rs_fixed'] = Rs_fixed
        outputs['ts_fixed'] = ts_fixed
        outputs['pts3D_pred_fixed'] = (similarity_mat @ pts3D_pred)  # 4,n
        outputs['pts3D_triangulated_fixed'] = (similarity_mat @ pts3D_triangulated)

        if bundle_adjustment:
            if BA_ERROR is not None:
                raise BA_ERROR
            repeat = conf.get_bool('ba.repeat')
            triangulation = conf.get_bool('ba.triangulation')
            ba_res = ba_functions.euc_ba(xs, Rs=Rs_pred, ts=ts_pred, Ks=np.linalg.inv(Ns),
                                         Xs_our=pts3D_pred.T, Ps=None,
                                         Ns=Ns, repeat=repeat, triangulation=triangulation, return_repro=True) #    Rs, ts, Ps, Xs
            outputs['Rs_ba'] = ba_res['Rs']
            outputs['ts_ba'] = ba_res['ts']
            outputs['Xs_ba'] = ba_res['Xs'].T  # 4,n
            outputs['Ps_ba'] = ba_res['Ps']

            R_ba_fixed, t_ba_fixed, similarity_mat = geo_utils.align_cameras(ba_res['Rs'], Rs_gt, ba_res['ts'], ts_gt,
                                                                       return_alignment=True)  # Align  Rs_fixed, tx_fixed
            outputs['Rs_ba_fixed'] = R_ba_fixed
            outputs['ts_ba_fixed'] = t_ba_fixed
            outputs['Xs_ba_fixed'] = (similarity_mat @ outputs['Xs_ba'])

    else:
        if bundle_adjustment:
            repeat = conf.get_bool('ba.repeat')
            triangulation = conf.get_bool('ba.triangulation')
            ba_res = ba_functions.proj_ba(Ps=Ps, xs=xs, Xs_our=pts3D_pred.T, Ns=Ns, repeat=repeat,
                                          triangulation=triangulation, return_repro=True, normalize_in_tri=True)   # Ps, Xs
            outputs['Xs_ba'] = ba_res['Xs'].T  # 4,n
            outputs['Ps_ba'] = ba_res['Ps']

    return outputs


def compute_errors(outputs, conf, bundle_adjustment):
    model_errors = {}
    calibrated = conf.get_bool('dataset.calibrated')
    Ps = outputs['Ps']
    pts3D_pred = outputs['pts3D_pred']
    xs = outputs['xs']
    pts3D_triangulated = outputs['pts3D_triangulated']

    model_errors["our_repro"] = np.nanmean(geo_utils.reprojection_error_with_points(Ps, pts3D_pred.T, xs))
    model_errors["triangulated_repro"] = np.nanmean(geo_utils.reprojection_error_with_points(Ps, pts3D_triangulated.T, xs))
    if calibrated:
        Rs_fixed = outputs['Rs_fixed']
        ts_fixed = outputs['ts_fixed']
        Rs_gt = outputs['Rs_gt']
        ts_gt = outputs['ts_gt']
        Rs_error, ts_error = geo_utils.tranlsation_rotation_errors(Rs_fixed, ts_fixed, Rs_gt, ts_gt)
        model_errors["ts_mean"] = np.mean(ts_error)
        model_errors["ts_med"] = np.median(ts_error)
        model_errors["Rs_mean"] = np.mean(Rs_error)
        model_errors["Rs_med"] = np.median(Rs_error)

    if bundle_adjustment:
        Xs_ba = outputs['Xs_ba']
        Ps_ba = outputs['Ps_ba']
        model_errors['repro_ba'] = np.nanmean(geo_utils.reprojection_error_with_points(Ps_ba, Xs_ba.T, xs))
        if calibrated:
            Rs_fixed = outputs['Rs_ba_fixed']
            ts_fixed = outputs['ts_ba_fixed']
            Rs_gt = outputs['Rs_gt']
            ts_gt = outputs['ts_gt']
            Rs_ba_error, ts_ba_error = geo_utils.tranlsation_rotation_errors(Rs_fixed, ts_fixed, Rs_gt, ts_gt)
            model_errors["ts_ba_mean"] = np.mean(ts_ba_error)
            model_errors["ts_ba_med"] = np.median(ts_ba_error)
            model_errors["Rs_ba_mean"] = np.mean(Rs_ba_error)
            model_errors["Rs_ba_med"] = np.median(Rs_ba_error)
    # Rs errors mean, ts errors mean, ba repro, rs ba mean, ts ba mean

    projected_pts = geo_utils.get_positive_projected_pts_mask(Ps @ pts3D_pred, conf.get_float('loss.infinity_pts_margin'))
    valid_pts = geo_utils.xs_valid_points(xs)
    unprojected_pts = np.logical_and(~projected_pts, valid_pts)
    part_unprojected = unprojected_pts.sum() / valid_pts.sum()

    model_errors['unprojected'] = part_unprojected

    return model_errors





