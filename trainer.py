import torch
from networks import VAE
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import weight_norm
import torch.distributions as dist
class Normal:
    def __init__(self, device='cpu', args=None):
        super().__init__()
        self.device = device
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'

    def sample(self, mu, v):
        eps = self._dist.sample(mu.size()).squeeze()
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        """compute the log-pdf of a normal distribution with diagonal covariance"""
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)
        lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf

class Trainer(nn.Module):
    def __init__(self, input_dim, n_domains, hidden_dim=32, device='cpu', args=None, latent_dim=None):
        super(Trainer, self).__init__()
        self.device = device
        self.args = args
        self.model = VAE(input_dim, n_domains, hidden_dim, latent_dim=latent_dim).to(device)
        self.n_domains = n_domains
        gate_params = [p for n, p in self.model.named_parameters() if 'adj_mat' in n]
        print(gate_params)
        nogate_params = [p for n, p in self.model.named_parameters() if n not in ['adj_mat']]

        self.optim = torch.optim.Adam(nogate_params, lr=1e-3)
        self.gate_optim = torch.optim.SGD(gate_params, lr=1e-3, momentum=0.9)
        self.input_dim = input_dim

    def decay_lr(self):
        return

    def step(self, x, d, target_kl, z):
        y = F.one_hot(d, self.n_domains).float()
        latent_dict = self.model(x, y)
        q_dist = dist.Normal(latent_dict['mu'], torch.exp(0.5*latent_dict['logvar']))
        z_dist = dist.Normal(latent_dict['cond_mean'], torch.exp(0.5*latent_dict['prior_log_var']))

        log_q_z_x = q_dist.log_prob(latent_dict['z'])
        log_p_z = z_dist.log_prob(latent_dict['z'])
        loss_kl = (log_q_z_x - log_p_z).sum(dim=-1).mean()

        loss_rec = F.mse_loss(latent_dict['xhat'], x)
        current_adj_mat = latent_dict['current_adj_mat']
        i_plus_a = torch.eye(len(current_adj_mat)).to(self.device) + current_adj_mat
        loss_sparsity = torch.matmul(i_plus_a.T, i_plus_a).sum()
        loss = self.args.lambda_kl*(loss_kl) + self.args.lambda_rec*loss_rec + self.args.lambda_sparse*loss_sparsity
        self.optim.zero_grad()
        self.gate_optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.gate_optim.step()
        return {'rec': loss_rec.item(), 'kl': loss_kl.item(), 'loss': loss.item(),  'target_kl': target_kl,
                'loss_sparse': loss_sparsity.item(),
                }


    @torch.no_grad()
    def evaluate(self, loader):
        zs = []
        true_zs = []
        num_samples = 0
        for x, z, d in loader:
            x, d = x.to(self.device), d.to(self.device)
            d = F.one_hot(d, self.n_domains).float()
            est_z = self.model(x, d)['mu']
            zs.append(est_z)
            true_zs.append(z)
            num_samples += x.size(0)
            if num_samples > 20000:
                break
        zs = torch.cat(zs)[:20000]
        true_zs = torch.cat(true_zs)[:20000]
        return zs, true_zs

    @torch.no_grad()
    def evaluate_all(self, loader):
        zs = []
        true_zs = []
        xs = []
        us = []
        num_samples = 0
        for x, z, d in loader:
            x, d = x.to(self.device), d.to(self.device)
            d = F.one_hot(d, self.n_domains).float()
            est_z = self.model(x, d)['mu']
            zs.append(est_z)
            true_zs.append(z)
            xs.append(x)
            us.append(d)
            num_samples += x.size(0)
        zs = torch.cat(zs)
        true_zs = torch.cat(true_zs)
        xs = torch.cat(xs)
        us = torch.cat(us)
        data_dict = {'our_z': zs, 'true_z': true_zs, 'x': xs, 'u': us}
        return data_dict
