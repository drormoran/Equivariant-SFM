import torch
from torch.nn import Linear, ReLU, BatchNorm1d, Sequential, Module, Identity
from utils.sparse_utils import SparseMat
from utils.pos_enc_utils import get_embedder


def get_linear_layers(feats, final_layer=False, batchnorm=True):
    layers = []

    # Add layers
    for i in range(len(feats) - 2):
        layers.append(Linear(feats[i], feats[i + 1]))

        if batchnorm:
            layers.append(BatchNorm1d(feats[i + 1], track_running_stats=False))

        layers.append(ReLU())

    # Add final layer
    layers.append(Linear(feats[-2], feats[-1]))
    if not final_layer:
        if batchnorm:
            layers.append(BatchNorm1d(feats[-1], track_running_stats=False))

        layers.append(ReLU())

    return Sequential(*layers)


class Parameter3DPts(torch.nn.Module):
    def __init__(self, n_pts):
        super().__init__()

        # Init points randomly
        pts_3d = torch.normal(mean=0, std=0.1, size=(3, n_pts), requires_grad=True)

        self.pts_3d = torch.nn.Parameter(pts_3d)

    def forward(self):
        return self.pts_3d


class SetOfSetLayer(Module):
    def __init__(self, d_in, d_out):
        super(SetOfSetLayer, self).__init__()
        # n is the number of points and m is the number of cameras
        self.lin_all = Linear(d_in, d_out)
        self.lin_n = Linear(d_in, d_out)
        self.lin_m = Linear(d_in, d_out)
        self.lin_both = Linear(d_in, d_out)

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        out_all = self.lin_all(x.values)  # [nnz,d_in] -> [nnz,d_out]

        mean_rows = x.mean(dim=0) # [m,n,d_in] -> [n,d_in]
        out_rows = self.lin_n(mean_rows)  # [n,d_in] -> [n,d_out]

        mean_cols = x.mean(dim=1) # [m,n,d_in] -> [m,d_in]
        out_cols = self.lin_m(mean_cols)  # [m,d_in] -> [m,d_out]

        out_both = self.lin_both(x.values.mean(dim=0, keepdim=True))  # [1,d_in] -> [1,d_out]

        new_features = (out_all + out_rows[x.indices[1], :] + out_cols[x.indices[0], :] + out_both) / 4  # [nnz,d_out]
        new_shape = (x.shape[0], x.shape[1], new_features.shape[1])

        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)


class ProjLayer(Module):
    def __init__(self, d_in, d_out):
        super(ProjLayer, self).__init__()
        # n is the number of points and m is the number of cameras
        self.lin_all = Linear(d_in, d_out)

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        new_features = self.lin_all(x.values)  # [nnz,d_in] -> [nnz,d_out]
        new_shape = (x.shape[0], x.shape[1], new_features.shape[1])
        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)


class NormalizationLayer(Module):
    def forward(self, x):
        features = x.values
        norm_features = features - features.mean(dim=0, keepdim=True)
        # norm_features = norm_features / norm_features.std(dim=0, keepdim=True)
        return SparseMat(norm_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)


class ActivationLayer(Module):
    def __init__(self):
        super(ActivationLayer, self).__init__()
        self.relu = ReLU()

    def forward(self, x):
        new_features = self.relu(x.values)
        return SparseMat(new_features, x.indices, x.cam_per_pts, x.pts_per_cam, x.shape)


class IdentityLayer(Module):
    def forward(self, x):
        return x


class EmbeddingLayer(Module):
    def __init__(self, multires, in_dim):
        super(EmbeddingLayer, self).__init__()
        if multires > 0:
            self.embed, self.d_out = get_embedder(multires, in_dim)
        else:
            self.embed, self.d_out = (Identity(), in_dim)

    def forward(self, x):
        embeded_features = self.embed(x.values)
        new_shape = (x.shape[0], x.shape[1], embeded_features.shape[1])
        return SparseMat(embeded_features, x.indices, x.cam_per_pts, x.pts_per_cam, new_shape)
