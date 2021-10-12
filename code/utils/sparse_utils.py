import torch


class SparseMat:
    def __init__(self, values, indices, cam_per_pts, pts_per_cam, shape):
        assert len(shape) == 3
        self.values = values
        self.indices = indices
        self.shape = shape
        self.cam_per_pts = cam_per_pts
        self.pts_per_cam = pts_per_cam
        self.device = self.values.device

    @property
    def size(self):
        return self.shape

    def sum(self, dim):
        assert dim == 1 or dim == 0
        n_features = self.shape[2]
        out_size = self.shape[0] if dim == 1 else self.shape[1]
        indices_index = 0 if dim == 1 else 1
        mat_sum = torch.zeros(out_size, n_features, device=self.device)
        return mat_sum.index_add(0, self.indices[indices_index], self.values)

    def mean(self, dim):
        assert dim == 1 or dim == 0
        if dim == 0:
            return self.sum(dim=0) / self.cam_per_pts
        else:
            return self.sum(dim=1) / self.pts_per_cam

    def to(self, device, **kwargs):
        self.device = device
        self.values = self.values.to(device, **kwargs)
        self.indices = self.indices.to(device, **kwargs)
        self.pts_per_cam = self.pts_per_cam.to(device, **kwargs)
        self.cam_per_pts = self.cam_per_pts.to(device, **kwargs)
        return self

    def __add__(self, other):
        assert self.shape == other.shape
        # assert (self.indices == other.indices).all()  # removed due to runtime
        new_values = self.values + other.values
        return SparseMat(new_values, self.indices, self.cam_per_pts, self.pts_per_cam, self.shape)