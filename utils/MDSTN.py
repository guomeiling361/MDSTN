import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 时间建模分支
class TemporalModelingBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3,
                 use_causal_conv=True, temporal_use_transformer=False):
        super().__init__()
        self.use_causal_conv = use_causal_conv
        self.temporal_use_transformer = temporal_use_transformer

        # 因果卷积
        if self.use_causal_conv:
            self.causal_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size - 1, bias=True)

        # 长期依赖建模
        if self.temporal_use_transformer:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, dropout=0.1,
                                           batch_first=True),
                num_layers=2
            )
        else:
            self.mamba = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)

        # 特征融合
        fusion_input_dim = hidden_dim * 2 if self.use_causal_conv else hidden_dim
        self.fusion = nn.Sequential(nn.Linear(fusion_input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        residual = self.residual_proj(x)
        if self.use_causal_conv:
            conv_input = x.transpose(1, 2)
            short_term = self.causal_conv(conv_input)[..., :x.shape[1]].transpose(1, 2)
            long_term = self.transformer(x) if self.temporal_use_transformer else self.mamba(x)
            fused = self.fusion(torch.cat([short_term, long_term], dim=-1))
        else:
            long_term = self.transformer(x) if self.temporal_use_transformer else self.mamba(x)
            fused = self.fusion(long_term)
        return fused + residual


# 空间建模分支
class SpatialModelingBranch(nn.Module):
    def __init__(self, input_channels, growth_rate=32, bn_size=4, drop_rate=0.2, d_model=64,
                 use_dense_conv=True, spatial_use_transformer=False):
        super().__init__()
        self.use_dense_conv = use_dense_conv
        self.spatial_use_transformer = spatial_use_transformer
        self.d_model = d_model

        # 局部空间特征
        if self.use_dense_conv:
            self.dense_block = _DenseBlock(4, input_channels, bn_size, growth_rate, drop_rate)
            self.transition = _Transition(input_channels + 4 * growth_rate, (input_channels + 4 * growth_rate) // 2)
            self.reduce_dim = (input_channels + 4 * growth_rate) // 2
        else:
            self.reduce_dim = d_model
            self.simple_conv = nn.Sequential(nn.Conv2d(input_channels, self.reduce_dim, 1),
                                             nn.BatchNorm2d(self.reduce_dim), nn.ReLU())

        # 全局空间特征
        if self.spatial_use_transformer:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.reduce_dim, nhead=8, dim_feedforward=self.reduce_dim * 4,
                                           dropout=0.1, batch_first=True),
                num_layers=2
            )
        else:
            self.mamba = Mamba(d_model=self.reduce_dim, d_state=16, d_conv=4, expand=2)

        self.fusion = nn.Sequential(nn.Conv2d(self.reduce_dim * 2, self.reduce_dim, 1), nn.BatchNorm2d(self.reduce_dim),
                                    nn.ReLU())

    def forward(self, x):
        # 局部特征
        local_feat = self.dense_block(x) if self.use_dense_conv else self.simple_conv(x)
        if self.use_dense_conv:
            local_feat = self.transition(local_feat)

        # 全局特征
        b, c, h, w = local_feat.shape
        seq_input = local_feat.view(b, c, h * w).transpose(1, 2)
        global_feat = self.transformer(seq_input) if self.spatial_use_transformer else self.mamba(seq_input)
        global_feat = global_feat.transpose(1, 2).view(b, c, h, w)

        return self.fusion(torch.cat([local_feat, global_feat], dim=1))


# 融合模块
class FusionModule(nn.Module):
    def __init__(self, feat_dim, fusion_mode=0):
        super().__init__()
        self.fusion_mode = fusion_mode
        if self.fusion_mode == 0:
            self.gate = nn.Sequential(nn.Linear(feat_dim * 2, feat_dim), nn.Sigmoid())
            self.fusion = nn.Linear(feat_dim * 3, feat_dim)
        else:
            self.fusion = nn.Sequential(nn.Linear(feat_dim * 2, feat_dim), nn.LayerNorm(feat_dim), nn.ReLU())

    def forward(self, st_feat, cross_feat):
        b, c, h, w = st_feat.shape
        st_flat = st_feat.view(b, c, -1).transpose(1, 2)
        cross_flat = cross_feat.view(b, c, -1).transpose(1, 2)

        if self.fusion_mode == 0:
            gate = self.gate(torch.cat([st_flat, cross_flat], dim=-1))
            fused_flat = gate * st_flat + (1 - gate) * cross_flat
            fused_flat = self.fusion(torch.cat([fused_flat, st_flat, cross_flat], dim=-1))
            return fused_flat.transpose(1, 2).view(b, c, h, w), gate
        else:
            fused_flat = self.fusion(torch.cat([st_flat, cross_flat], dim=-1))
            return fused_flat.transpose(1, 2).view(b, c, h, w), None


# 密集网络组件
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, 1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x.contiguous())
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, 1, bias=False))


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module(f'denselayer{i + 1}', layer)


# 主模型 MCDAG
class MCDAG(nn.Module):
    def __init__(self, input_shape, meta_shape, cross_shape, nb_flows,
                 use_causal_conv=True, temporal_use_transformer=False,
                 use_dense_conv=True, spatial_use_transformer=False,
                 fusion_mode=0):
        super().__init__()
        self.seq_len = input_shape[1]
        self.H, self.W = input_shape[3], input_shape[4]
        self.hist_channels = input_shape[2]
        self.feat_dim = 64

        # 功能开关
        self.use_meta = meta_shape is not None
        self.use_cross = cross_shape is not None

        # 嵌入层
        self.hist_embedding = nn.Sequential(nn.Linear(self.hist_channels, self.feat_dim), nn.LayerNorm(self.feat_dim),
                                            nn.ReLU())

        # 跨域特征
        if self.use_cross:
            self.cross_embedding = nn.Sequential(nn.Conv2d(cross_shape[1], self.feat_dim, 1),
                                                 nn.BatchNorm2d(self.feat_dim), nn.ReLU())
            self.cross_encoder = nn.Sequential(_DenseBlock(2, self.feat_dim, 2, 8, 0.2),
                                               _Transition(self.feat_dim + 16, self.feat_dim))
            self.st_cross_fusion = FusionModule(self.feat_dim, fusion_mode)

        # 元数据
        if self.use_meta:
            self.meta_embedding = nn.Sequential(nn.Linear(meta_shape[1], self.feat_dim), nn.LayerNorm(self.feat_dim),
                                                nn.ReLU(), nn.Dropout(0.3))

        # 时空分支
        self.temporal_branch = TemporalModelingBranch(self.feat_dim, self.feat_dim, 3, use_causal_conv,
                                                      temporal_use_transformer)
        self.spatial_branch = SpatialModelingBranch(self.feat_dim, 16, 4, 0.2, self.feat_dim, use_dense_conv,
                                                    spatial_use_transformer)

        # 融合层
        self.st_integration = nn.Sequential(nn.Conv2d(self.feat_dim * 2, self.feat_dim, 1),
                                            nn.BatchNorm2d(self.feat_dim), nn.ReLU())
        final_fusion_input = self.feat_dim * 2 if self.use_meta else self.feat_dim
        self.final_fusion = nn.Sequential(nn.Conv2d(final_fusion_input, self.feat_dim, 1),
                                          nn.BatchNorm2d(self.feat_dim), nn.ReLU())

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.feat_dim // 2, 3, padding=1), nn.BatchNorm2d(self.feat_dim // 2), nn.ReLU(),
            nn.Conv2d(self.feat_dim // 2, nb_flows, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, cross_data, meta_data):
        batch_size = x.shape[0]
        # 历史特征嵌入
        x_reshaped = x.permute(0, 1, 3, 4, 2)
        hist_feat = self.hist_embedding(x_reshaped)

        # 跨域/元数据处理
        cross_feat = self.cross_embedding(cross_data) if self.use_cross else None
        meta_feat = self.meta_embedding(meta_data).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.H,
                                                                                      self.W) if self.use_meta else None

        # 时间特征
        temporal_input = hist_feat.view(batch_size, self.seq_len, -1, self.feat_dim).transpose(2, 1).reshape(-1,
                                                                                                             self.seq_len,
                                                                                                             self.feat_dim)
        temporal_feat = self.temporal_branch(temporal_input)[:, -1, :].view(batch_size, self.H, self.W,
                                                                            self.feat_dim).permute(0, 3, 1, 2)

        # 空间特征
        spatial_feat = self.spatial_branch(hist_feat[:, -1].permute(0, 3, 1, 2))

        # 时空融合
        st_feat = self.st_integration(torch.cat([temporal_feat, spatial_feat], dim=1))

        # 跨域融合
        if self.use_cross:
            cross_encoded = self.cross_encoder(cross_feat)
            st_cross_feat, gate_z = self.st_cross_fusion(st_feat, cross_encoded)
        else:
            st_cross_feat, gate_z = st_feat, None

        # 最终融合
        final_feat = self.final_fusion(
            torch.cat([st_cross_feat, meta_feat], dim=1)) if self.use_meta else self.final_fusion(st_cross_feat)
        return torch.sigmoid(self.output_layer(final_feat)), gate_z
