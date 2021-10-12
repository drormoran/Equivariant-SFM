import abc

import torch
from utils import geo_utils
from pytorch3d import transforms as py3d_trans


class BaseNet(torch.nn.Module):
    def __init__(self, conf):
        super(BaseNet, self).__init__()

        self.calibrated = conf.get_bool('dataset.calibrated')
        self.normalize_output = conf.get_string('model.normalize_output', default=None)
        self.rot_representation = conf.get_string('model.rot_representation', default='quat')
        self.soft_sign = torch.nn.Softsign()

        if self.calibrated and self.rot_representation == '6d':
            print('rot representation: ' + self.rot_representation)
            self.out_channels = 9
        elif self.calibrated and self.rot_representation == 'quat':
            self.out_channels = 7
        elif self.calibrated and self.rot_representation == 'svd':
            self.out_channels = 12
        elif not self.calibrated:
            self.out_channels = 12
        else:
            print("Illegal output format")
            exit()

    @abc.abstractmethod
    def forward(self, data):
        pass

    def extract_model_outputs(self, x, pts_3D, data):
        # Get points
        pts_3D = geo_utils.ones_padding(pts_3D)

        # Get calibrated predictions
        if self.calibrated:
            # Get rotation
            if self.rot_representation == '6d':
                RTs = py3d_trans.rotation_6d_to_matrix(x[:, :6])
            elif self.rot_representation == 'svd':
                m = x[:, :9].reshape(-1, 3, 3)
                RTs = geo_utils.project_to_rot(m)
            elif self.rot_representation == 'quat':
                RTs = py3d_trans.quaternion_to_matrix(x[:, :4])
            else:
                print("Illegal output format")
                exit()

            # Get translation
            minRTts = x[:, -3:]

            # Get camera matrix
            Ps = torch.cat((RTs, minRTts.unsqueeze(dim=-1)), dim=-1)

        else:  # Projective
            Ps = x.reshape(-1, 3, 4)

            # Normalize predictions
            if self.normalize_output == "Chirality":
                scale = torch.sign(Ps[:, 0:3, 0:3].det()) / Ps[:, 2, 0:3].norm(dim=1)
                Ps = Ps * scale.reshape(-1, 1, 1)
            elif self.normalize_output == "Differentiable Chirality":
                scale = self.soft_sign(Ps[:, 0:3, 0:3].det() * 10e3) / Ps[:, 2, 0:3].norm(dim=1)
                Ps = Ps * scale.reshape(-1, 1, 1)
            elif self.normalize_output == "Frobenius":
                Ps = Ps / Ps.norm(dim=(1, 2), p='fro', keepdim=True)

        # The model outputs a normalized camera! Meaning from world coordinates to camera coordinates, not to pixels in the image.
        pred_cams = {"Ps_norm": Ps, "pts3D": pts_3D}
        return pred_cams

