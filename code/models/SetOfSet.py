import torch
from torch import nn
from models.baseNet import BaseNet
from models.layers import *


class SetOfSetBlock(nn.Module):
    def __init__(self, d_in, d_out, conf):
        super(SetOfSetBlock, self).__init__()
        self.block_size = conf.get_int("model.block_size")
        self.use_skip = conf.get_bool("model.use_skip")

        modules = []
        modules.extend([SetOfSetLayer(d_in, d_out), NormalizationLayer()])
        for i in range(1, self.block_size):
            modules.extend([ActivationLayer(), SetOfSetLayer(d_out, d_out), NormalizationLayer()])
        self.layers = nn.Sequential(*modules)

        self.final_act = ActivationLayer()

        if self.use_skip:
            if d_in == d_out:
                self.skip = IdentityLayer()
            else:
                self.skip = nn.Sequential(ProjLayer(d_in, d_out), NormalizationLayer())

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        xl = self.layers(x)
        if self.use_skip:
            xl = self.skip(x) + xl

        out = self.final_act(xl)
        return out


class SetOfSetNet(BaseNet):
    def __init__(self, conf):
        super(SetOfSetNet, self).__init__(conf)
        # n is the number of points and m is the number of cameras
        num_blocks = conf.get_int('model.num_blocks')
        num_feats = conf.get_int('model.num_features')
        multires = conf.get_int('model.multires')

        n_d_out = 3
        m_d_out = self.out_channels
        d_in = 2

        self.embed = EmbeddingLayer(multires, d_in)

        self.equivariant_blocks = torch.nn.ModuleList([SetOfSetBlock(self.embed.d_out, num_feats, conf)])
        for i in range(num_blocks - 1):
            self.equivariant_blocks.append(SetOfSetBlock(num_feats, num_feats, conf))

        self.m_net = get_linear_layers([num_feats] * 2 + [m_d_out], final_layer=True, batchnorm=False)
        self.n_net = get_linear_layers([num_feats] * 2 + [n_d_out], final_layer=True, batchnorm=False)

    def forward(self, data):
        x = data.x  # x is [m,n,d] sparse matrix
        x = self.embed(x)
        for eq_block in self.equivariant_blocks:
            x = eq_block(x)  # [m,n,d_in] -> [m,n,d_out]

        # Cameras predictions
        m_input = x.mean(dim=1) # [m,d_out]
        m_out = self.m_net(m_input)  # [m, d_m]

        # Points predictions
        n_input = x.mean(dim=0) # [n,d_out]
        n_out = self.n_net(n_input).T  # [n, d_n] -> [d_n, n]

        pred_cam = self.extract_model_outputs(m_out, n_out, data)

        return pred_cam




