# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from bimanual_imitation.algorithms.configs import ActParamConfig

from .transformer import TransformerEncoder, TransformerEncoderLayer, build_transformer


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """This is the DETR module that performs object detection"""

    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        latent_dim,
        num_queries,
        action_dim_single,
        obs_dim,
    ):
        """Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        assert backbones is None, "Backbone not used/supported for bimanual setup!"
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model

        self.action_head = nn.Linear(hidden_dim, action_dim_single)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj_robot_state = nn.Linear(obs_dim, hidden_dim)
        self.pos = torch.nn.Embedding(2, hidden_dim)

        # encoder extra parameters
        self.latent_dim = latent_dim
        # extra cls token embedding
        self.cls_embed = nn.Embedding(1, hidden_dim)
        # project action to embedding
        self.encoder_action_proj = nn.Linear(action_dim_single, hidden_dim)
        # project qpos to embedding
        self.encoder_joint_proj = nn.Linear(obs_dim, hidden_dim)
        # project hidden state to latent std, var
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        # [CLS], qpos, a_seq
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
        )
        # project latent sample to embedding
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        # learned position embedding for proprio and latent
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def forward(self, cur_obs, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        bs, _ = cur_obs.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            cur_obs_embed = self.encoder_joint_proj(cur_obs)  # (bs, hidden_dim)
            cur_obs_embed = torch.unsqueeze(cur_obs_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
            # (bs, seq+1, hidden_dim)
            encoder_input = torch.cat([cls_embed, cur_obs_embed, action_embed], axis=1)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(cur_obs.device)  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                cur_obs.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        proprio_input = self.input_proj_robot_state(cur_obs)
        hs = self.transformer(
            src=None,
            mask=None,
            query_embed=self.query_embed.weight,
            pos_embed=self.pos.weight,
            latent_input=latent_input,
            proprio_input=proprio_input,
            additional_pos_embed=self.additional_pos_embed.weight,
        )[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(cfg: ActParamConfig):
    encoder_layer = TransformerEncoderLayer(
        d_model=cfg.hidden_dim,
        nhead=cfg.n_heads,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        activation=cfg.activation,
        normalize_before=cfg.pre_norm,
    )
    encoder_norm = nn.LayerNorm(cfg.hidden_dim) if cfg.pre_norm else None
    encoder = TransformerEncoder(
        encoder_layer=encoder_layer, num_layers=cfg.enc_layers, norm=encoder_norm
    )

    return encoder


def build(cfg: ActParamConfig, action_dim_single, obs_dim):
    # NOTE: backbone is not used/needed for bimanual (non-image observation) setup
    backbones = None

    transformer = build_transformer(cfg)
    encoder = build_encoder(cfg)

    model = DETRVAE(
        backbones=backbones,
        transformer=transformer,
        encoder=encoder,
        latent_dim=cfg.latent_dim,
        num_queries=cfg.pred_horizon,
        action_dim_single=action_dim_single,
        obs_dim=obs_dim,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[ACT] Number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model
