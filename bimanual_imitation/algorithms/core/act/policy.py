import torch
import torch.nn as nn
from torch.nn import functional as F

from bimanual_imitation.algorithms.configs import ActParamConfig
from bimanual_imitation.algorithms.core.act.detr_vae import build as build_vae


class ACTPolicy(nn.Module):
    def __init__(self, cfg: ActParamConfig, action_dim_single, obs_dim):
        super().__init__()

        self.model = build_vae(cfg, action_dim_single, obs_dim)

        # NOTE: backbone is not used/needed for bimanual (non-image observation) setup
        for n, p in self.model.named_parameters():
            assert "backbone" not in n, "Backbone not used/supported for bimanual setup!"

        param_dicts = [{"params": [p for n, p in self.model.named_parameters() if p.requires_grad]}]
        self.optimizer = torch.optim.AdamW(
            param_dicts, lr=cfg.act_lr, weight_decay=cfg.weight_decay
        )
        self.kl_weight = cfg.kl_weight

    def __call__(self, cur_obs, actions=None, is_pad=None):
        if actions is not None:  # training time
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(cur_obs, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(cur_obs)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
