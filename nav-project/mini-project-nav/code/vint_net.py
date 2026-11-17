"""The ViNT network, in plain PyTorch.

This is the model of "ViNT: A Foundation Model for Visual Navigation" (Shah et al., 2023), copied from the authors' repository
(https://github.com/robodhruv/visualnav-transformer, MIT license, train/vint_train/models/vint/) and trimmed to the one configuration
that vint.pt holds: six frames of context (5 past + the current one), an EfficientNet-B0 image encoder, a four-layer transformer,
and five predicted waypoints. Layer names are the authors' own, so their checkpoints load into it.
It is an ordinary nn.Module: you can add layers to it, freeze parts of it, and export it with export_onnx.export_module.

    obs   [B, 18, 64, 85]   six RGB frames, oldest first, stacked on the channel axis (the last one is the current frame)
    goal  [B, 3, 64, 85]    the goal picture
    returns (distance, action): the predicted temporal distance to the goal, [B, 1], and [B, 5, 4] with, per waypoint, the position
    (x forward, y left) in units of STEP meters, and the cosine and sine of the heading.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pos_enc", pos_enc.unsqueeze(0))

    def forward(self, x):
        return x + self.pos_enc[:, :x.size(1), :]


class MultiLayerDecoder(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=(256, 128, 64), nhead=8, num_layers=8, ff_dim_factor=4):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor * embed_dim,
                                                   activation="gelu", dropout=0.0, batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_layers = nn.ModuleList([nn.Linear(seq_len * embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers) - 1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i + 1]))

    def forward(self, x):
        x = self.sa_decoder(self.positional_encoding(x))                    # [B, seq_len, embed_dim]
        x = x.reshape(x.shape[0], -1)
        for layer in self.output_layers:
            x = F.relu(layer(x))
        return x


class ViNT(nn.Module):
    def __init__(self, context_size=5, len_traj_pred=5, obs_encoding_size=512, heads=4, layers=4, ff_dim_factor=4):
        super().__init__()
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.obs_encoding_size = obs_encoding_size
        self.obs_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3, drop_connect_rate=0.0)             # encodes each frame by itself
        self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6, drop_connect_rate=0.0)            # encodes the current frame and the goal together
        features = self.obs_encoder._fc.in_features
        self.compress_obs_enc = nn.Linear(features, obs_encoding_size)
        self.compress_goal_enc = nn.Linear(features, obs_encoding_size)
        self.decoder = MultiLayerDecoder(embed_dim=obs_encoding_size, seq_len=context_size + 2, output_layers=[256, 128, 64, 32],
                                         nhead=heads, num_layers=layers, ff_dim_factor=ff_dim_factor)
        self.dist_predictor = nn.Sequential(nn.Linear(32, 1))
        self.action_predictor = nn.Sequential(nn.Linear(32, len_traj_pred * 4))

    def train(self, mode=True):
        """Training mode as usual, except that the BatchNorm layers always use the statistics stored in vint.pt instead of updating them.
        The pretrained model was exported with those statistics fixed. Letting them follow the real-robot frames in training makes the
        model worse on the simulator's rendered frames, which look different."""
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def encode(self, encoder, images):
        """EfficientNet features of [N, C, H, W] pictures, pooled to [N, 1280]."""
        return encoder._avg_pooling(encoder.extract_features(images)).flatten(start_dim=1)

    def forward(self, obs, goal):
        current = obs[:, 3 * self.context_size:]                                            # the newest of the six frames
        goal_encoding = self.compress_goal_enc(self.encode(self.goal_encoder, torch.cat([current, goal], dim=1))).unsqueeze(1)
        frames = torch.cat(torch.split(obs, 3, dim=1), dim=0)                               # [6 * B, 3, H, W], frame-major
        obs_encoding = self.compress_obs_enc(self.encode(self.obs_encoder, frames))
        obs_encoding = obs_encoding.reshape(self.context_size + 1, -1, self.obs_encoding_size).transpose(0, 1)      # [B, 6, 512]
        final = self.decoder(torch.cat([obs_encoding, goal_encoding], dim=1))               # [B, 32]
        action = self.action_predictor(final).reshape(len(obs), self.len_traj_pred, 4)
        position = torch.cumsum(action[..., :2], dim=1)                                     # the network predicts steps; add them up
        heading = F.normalize(action[..., 2:], dim=-1)
        return self.dist_predictor(final), torch.cat([position, heading], dim=-1)
