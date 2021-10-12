import plotly
import os
import utils.path_utils
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import torch
import plotly.graph_objects as go
from utils import geo_utils
from matplotlib import image


def plot_img_sets_error_bar(err_list, imgs_sets, path, title):
    max_val = 20

    # Prepare illegal values
    #illegal_idx = (err_list == float("inf"))
    illegal_idx = torch.logical_or((err_list > max_val), (err_list == float("inf")))
    final_err = err_list.clone()
    final_err[illegal_idx] = max_val
    colors = np.array(['#636efa', ] * final_err.shape[0])
    colors[illegal_idx] = 'crimson'

    colors = colors.tolist()
    final_err = final_err.tolist()

    # Create figure
    fig = px.bar(x=imgs_sets, y=final_err)
    fig.update_xaxes(title='Images sets')
    fig.update_yaxes(title=title, range=[0, max_val])
    fig.update_traces(marker_color=colors)
    fig.update_layout(xaxis_type='category')

    plotly.offline.plot(fig, filename=path)


def plot_img_reprojection_error_bar(err_list, img_list):
    return go.Bar(x=img_list, y=err_list.tolist())
    # max_val = conf.get_int('plot.reproj_err_bar_max', default=20)
    #
    # # Create figure
    # fig = go.Figure([go.Bar(x=img_list.tolist(), y=err_list.tolist())])
    # fig.update_xaxes(title='Images')
    # fig.update_yaxes(title='Reprojection Error', range=[0, max_val])
    # # fig.update_traces(marker_color='crimson')
    # fig.update_layout(xaxis_type='category')
    #
    # path = os.path.join(general_utils.path_to_exp(conf), 'reprojection_err.html')
    # plotly.offline.plot(fig, filename=path)


def plot_error_per_images_bar(repreoj_err, symetric_epipolar_dist, imgs_sets, conf, sub_name=""):
    path = os.path.join(utils.path_utils.path_to_exp(conf), 'reprojection_err' + sub_name)
    plot_img_sets_error_bar(repreoj_err, imgs_sets, path, 'Mean Reprojection Error')

    path = os.path.join(utils.path_utils.path_to_exp(conf), 'SymEpDist' + sub_name)
    plot_img_sets_error_bar(symetric_epipolar_dist, imgs_sets, path, 'Mean Symmetric Epipolar Distance')


def plot_matrix_heatmap(data_matrix, indices, zmax):
    mask = data_matrix == 0
    mat = data_matrix.clone().numpy()
    mat[mask] = None
    #fig = px.imshow(mat, x=indices, y=indices)
    hm = go.Heatmap(z=mat, x=indices, y=indices, zmin=0, zmax=zmax)

    return hm


def plot_heatmaps(repreoj_err, symetric_epipolar_dist, global_reprojection_error, edges, img_list, conf, path=None, static_path=None):
    repreoj_err_edges = torch.zeros(repreoj_err.shape)
    repreoj_err_edges[edges[0], edges[1]] = repreoj_err[edges[0], edges[1]]

    symetric_epipolar_dist_edges = torch.zeros(symetric_epipolar_dist.shape)
    symetric_epipolar_dist_edges[edges[0], edges[1]] = symetric_epipolar_dist[edges[0], edges[1]]

    zmax = conf.get_int('plot.color_bar_max', default=5)
    hm_rep_err = plot_matrix_heatmap(repreoj_err, list(map(str, img_list)), zmax)
    hm_rep_err_edges = plot_matrix_heatmap(repreoj_err_edges, list(map(str, img_list)), zmax)

    hm_sed = plot_matrix_heatmap(symetric_epipolar_dist, list(map(str, img_list)), zmax)
    hm_sed_edges = plot_matrix_heatmap(symetric_epipolar_dist_edges, list(map(str, img_list)), zmax)

    bar_global_rep_err = plot_img_reprojection_error_bar(global_reprojection_error, img_list)

    fig = make_subplots(2, 4, subplot_titles=['Reprojection Error', 'Symmetric Epipolar Distance',
                                              'Reprojection Error - Triplets', 'Symmetric Epipolar Distance - Triplets',
                                              'Global Reprojection Error'],
                        specs=[[{}, {}, {}, {}], [{"colspan": 4}, None, None, None]])

    fig.add_trace(hm_rep_err, 1, 1)
    fig.update_xaxes(type='category', title='Image', row=1, col=1)
    fig.update_yaxes(type='category', title='Image', row=1, col=1)

    fig.add_trace(hm_sed, 1, 2)
    fig.update_xaxes(type='category', title='Image', row=1, col=2)
    fig.update_yaxes(type='category', title='Image', row=1, col=2)

    fig.add_trace(hm_rep_err_edges, 1, 3)
    fig.update_xaxes(type='category', title='Image', row=1, col=3)
    fig.update_yaxes(type='category', title='Image', row=1, col=3)

    fig.add_trace(hm_sed_edges, 1, 4)
    fig.update_xaxes(type='category', title='Image', row=1, col=4)
    fig.update_yaxes(type='category', title='Image', row=1, col=4)

    fig.add_trace(bar_global_rep_err, 2, 1)
    max_rep_err = conf.get_int('plot.reproj_err_bar_max', default=20)
    fig.update_xaxes(type='category', title='Image', row=2, col=1)
    fig.update_yaxes(title='Reprojection Error', range=[0, max_rep_err], row=2, col=1)

    # fig.update_layout(width=1000)
    if path is None:
        path = os.path.join(utils.path_utils.path_to_exp(conf), 'errors_heatmap.html')
    plotly.offline.plot(fig, filename=path)
    if static_path is not None:
        fig.write_image(static_path)


def plot_cameras_before_and_after_ba(outputs, errors, conf, phase, scan, epoch=None, bundle_adjustment=False):
    Rs_gt = outputs['Rs_gt']
    ts_gt = outputs['ts_gt']

    Rs_pred = outputs['Rs_fixed']
    ts_pred = outputs['ts_fixed']
    pts3D = outputs['pts3D_pred_fixed'][:3,:]
    Rs_error = errors['Rs_mean']
    ts_error = errors['ts_mean']
    plot_cameras(Rs_pred, ts_pred, pts3D, Rs_gt, ts_gt, Rs_error, ts_error, conf, phase, scan=scan, epoch=epoch)

    if bundle_adjustment:
        Rs_pred = outputs['Rs_ba_fixed']
        ts_pred = outputs['ts_ba_fixed']
        pts3D = outputs['Xs_ba_fixed'][:3,:]
        Rs_error = errors['Rs_ba_mean']
        ts_error = errors['ts_ba_mean']
        plot_cameras(Rs_pred, ts_pred, pts3D, Rs_gt, ts_gt, Rs_error, ts_error, conf, phase, scan=scan+'_ba', epoch=epoch)

def get_points_colors(images_path, image_names, xs, first_occurence=False):
    m, n, _ = xs.shape
    points_colors = np.zeros([n, 3])
    if first_occurence:
        images_indices = (geo_utils.xs_valid_points(xs)).argmax(axis=0)
        unique_images = np.unique(images_indices)
        for i, image_ind in enumerate(unique_images):
            image_name = str(image_names[image_ind][0]).split('/')[1]
            im = image.imread(os.path.join(images_path, image_name))
            # read the image to ndarray
            points_in_image = np.where(image_ind == images_indices)[0]
            for point_ind in points_in_image:
                point_2d_in_image = xs[image_ind, point_ind].astype(int)
                points_colors[point_ind] = im[point_2d_in_image[1], point_2d_in_image[0]]
    else:
        valid_points = geo_utils.xs_valid_points(xs)
        colors = np.zeros([m, n, 3])
        for image_ind in range(m):
            image_name = str(image_names[image_ind][0]).split('/')[1]
            im = image.imread(os.path.join(images_path, image_name))
            points_in_image = np.where(valid_points[image_ind])[0]
            for point_ind in points_in_image:
                point_2d_in_image = xs[image_ind, point_ind].astype(int)
                colors[image_ind, point_ind] = im[point_2d_in_image[1], point_2d_in_image[0]]
        for point_ind in range(n):
            points_colors[point_ind] = np.mean(colors[valid_points[:, point_ind], point_ind], axis=0)

    return points_colors

def plot_cameras(Rs_pred, ts_pred, pts3D, Rs_gt, ts_gt, Rs_error, ts_error, conf, phase, scan=None, epoch=None):
    data = []
    data.append(get_3D_quiver_trace(ts_gt, Rs_gt[:, :3, 2], color='#86CE00', name='cam_gt'))
    data.append(get_3D_quiver_trace(ts_pred, Rs_pred[:, :3, 2], color='#C4451C', name='cam_learn'))
    data.append(get_3D_scater_trace(ts_gt.T, color='#86CE00', name='cam_gt', size=1))
    data.append(get_3D_scater_trace(ts_pred.T, color='#C4451C', name='cam_learn', size=1))
    data.append(get_3D_scater_trace(pts3D, '#3366CC', '3D points', size=0.5))

    fig = go.Figure(data=data)
    fig.update_layout(title='Cameras: Rotation Mean = {:.5f}, Translation Mean = {:.5f}'.format(Rs_error.mean(), ts_error.mean()), showlegend=True)

    path = utils.path_utils.path_to_plots(conf, phase, epoch=epoch, scan=scan)
    plotly.offline.plot(fig, filename=path, auto_open=False)

    return path


def get_3D_quiver_trace(points, directions, color='#bd1540', name='', cam_size=1):
    assert points.shape[1] == 3, "3d cone plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d cone plot input points are not correctely shaped "
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctely shaped "
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctely shaped "

    trace = go.Cone(
        name=name,
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        u=directions[:, 0],
        v=directions[:, 1],
        w=directions[:, 2],
        sizemode='absolute',
        sizeref=cam_size,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        anchor="tail"
    )

    return trace


def get_3D_scater_trace(points, color, name,size=0.5):
    assert points.shape[0] == 3, "3d plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d plot input points are not correctely shaped "

    trace = go.Scatter3d(
        name=name,
        x=points[0, :],
        y=points[1, :],
        z=points[2, :],
        mode='markers',
        marker=dict(
            size=size,
            color=color,
        )
    )

    return trace


