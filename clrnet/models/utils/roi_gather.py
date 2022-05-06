import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


def LinearModule(hidden_dim):
    return nn.ModuleList(
        [nn.Linear(hidden_dim, hidden_dim),
         nn.ReLU(inplace=True)])


class FeatureResize(nn.Module):
    def __init__(self, size=(10, 25)):
        super(FeatureResize, self).__init__()
        self.size = size

    def forward(self, x):
        x = F.interpolate(x, self.size)
        return x.flatten(2)


class ROIGather(nn.Module):
    '''
    ROIGather module for gather global information
    Args: 
        in_channels: prior feature channels
        num_priors: prior numbers we predefined
        sample_points: the number of sampled points when we extract feature from line
        fc_hidden_dim: the fc output channel
        refine_layers: the total number of layers to build refine
    '''
    def __init__(self,
                 in_channels,
                 num_priors,
                 sample_points,
                 fc_hidden_dim,
                 refine_layers,
                 mid_channels=48):
        super(ROIGather, self).__init__()
        self.in_channels = in_channels
        self.num_priors = num_priors
        self.f_key = ConvModule(in_channels=self.in_channels,
                                out_channels=self.in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                norm_cfg=dict(type='BN'))

        self.f_query = nn.Sequential(
            nn.Conv1d(in_channels=num_priors,
                      out_channels=num_priors,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=num_priors),
            nn.ReLU(),
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.W = nn.Conv1d(in_channels=num_priors,
                           out_channels=num_priors,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=num_priors)

        self.resize = FeatureResize()
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.convs = nn.ModuleList()
        self.catconv = nn.ModuleList()
        for i in range(refine_layers):
            self.convs.append(
                ConvModule(in_channels,
                           mid_channels, (9, 1),
                           padding=(4, 0),
                           bias=False,
                           norm_cfg=dict(type='BN')))

            self.catconv.append(
                ConvModule(mid_channels * (i + 1),
                           in_channels, (9, 1),
                           padding=(4, 0),
                           bias=False,
                           norm_cfg=dict(type='BN')))

        self.fc = nn.Linear(sample_points * fc_hidden_dim, fc_hidden_dim)

        self.fc_norm = nn.LayerNorm(fc_hidden_dim)

    def roi_fea(self, x, layer_index):
        feats = []
        for i, feature in enumerate(x):
            feat_trans = self.convs[i](feature)
            feats.append(feat_trans)
        cat_feat = torch.cat(feats, dim=1)
        cat_feat = self.catconv[layer_index](cat_feat)
        return cat_feat

    def forward(self, roi_features, x, layer_index):
        '''
        Args:
            roi_features: prior feature, shape: (Batch * num_priors, prior_feat_channel, sample_point, 1)
            x: feature map
            layer_index: currently on which layer to refine
        Return: 
            roi: prior features with gathered global information, shape: (Batch, num_priors, fc_hidden_dim)
        '''
        roi = self.roi_fea(roi_features, layer_index)
        bs = x.size(0)
        roi = roi.contiguous().view(bs * self.num_priors, -1)

        roi = F.relu(self.fc_norm(self.fc(roi)))
        roi = roi.view(bs, self.num_priors, -1)
        query = roi

        value = self.resize(self.f_value(x))
        query = self.f_query(query)
        key = self.f_key(x)
        value = value.permute(0, 2, 1)
        key = self.resize(key)
        sim_map = torch.matmul(query, key)
        sim_map = (self.in_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = self.W(context)

        roi = roi + F.dropout(context, p=0.1, training=self.training)

        return roi
