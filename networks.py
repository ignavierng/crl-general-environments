import numpy as np
import pickle
import math
from sys import exit
import matplotlib.pyplot as plt
from sklearn import datasets
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
from torch.nn.utils import weight_norm


def weights_init(init_type='gaussian', gain=math.sqrt(2)):
	def init_fun(m):
		classname = m.__class__.__name__
		if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight') and (
				'Norm' not in classname):
			# print m.__class__.__name__
			if init_type == 'gaussian':
				init.normal_(m.weight.data, 0.0, 0.02)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=0.02)
			elif init_type == 'xavier_uniform':
				init.xavier_uniform_(m.weight.data)
			elif init_type == 'kaiming':
				init.kaiming_uniform_(m.weight.data)
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=math.sqrt(2))
			elif init_type == 'default':
				pass
			else:
				assert 0, "Unsupported initialization: {}".format(init_type)
		if hasattr(m, 'bias') and m.bias is not None:
			init.constant_(m.bias.data, 0.0)

	return init_fun


def sample_gumbel(shape, eps=1e-20):
	"""Sample from Gumbel(0, 1)"""
	U = torch.rand(shape)
	return -torch.log(-torch.log(U + eps) + eps)


def gumbel_sigmoid(logits, tau=1.0, hard=False, is_train=True):
	"""
	Gumbel-Sigmoid sampling.

	Args:
		logits: Input logits of shape [batch_size, d] (unnormalized probabilities).
		tau: Non-negative scalar temperature.
		hard: If True, the output will be discretized as 0 or 1, but the gradient will still be computed.

	Returns:
		Tensor of shape [batch_size, d], where values are between 0 and 1.
	"""
	# Sample Gumbel noise for both classes (0 and 1)
	gumbel_noise_1 = sample_gumbel(logits.size())
	gumbel_noise_2 = sample_gumbel(logits.size())

	# Compute the Gumbel-Sigmoid (continuous relaxation)
	y_soft = logits

	if hard:
		# Apply the straight-through estimator to make the output binary, but still differentiate through soft y
		y_hard = (y_soft > 0.5).float()
		y = (y_hard - y_soft).detach() + y_soft  # This keeps the gradient but returns hard output
		return y
	else:
		# Return the soft (continuous) output
		return y_soft


class ADJMatrix(nn.Module):
	def __init__(self, input_dim):
		super().__init__()
		self.adj_matrix = nn.Parameter(torch.ones([input_dim, input_dim]))

	def forward(self, soft=False):
		if soft:
			out = self.adj_matrix
		else:
			out = (self.adj_matrix > 0.5).float() - self.adj_matrix.detach() + self.adj_matrix
		out = torch.tril(out, diagonal=-1)
		return out


import functools


class ToConvShape(nn.Module):
	def __init__(self, c, h, w):
		super().__init__()
		self.c = c
		self.h = h
		self.w = w

	def forward(self, x):
		return x.view(len(x), self.c, self.h, self.w)


class VAE(nn.Module):
	def __init__(self, input_dim, n_domains, hidden_dim, latent_dim=None):
		super().__init__()
		if latent_dim is None:
			latent_dim = input_dim
		self.latent_dim = latent_dim
		ACT = functools.partial(nn.LeakyReLU, negative_slope=0.2)
		# ACT = nn.ELU
		# NORM = functools.partial(nn.LayerNorm, normalized_shape=hidden_dim)
		NORM = nn.LayerNorm
		self.encoder_mu = nn.Sequential(nn.Linear(input_dim, hidden_dim), ACT(),
		                                nn.Linear(hidden_dim, latent_dim)
		                             )
		self.encoder_logvar = nn.Sequential(nn.Linear(input_dim, hidden_dim), ACT(),
		                                    nn.Linear(hidden_dim, hidden_dim), NORM(hidden_dim), ACT(),
		                                    nn.Linear(hidden_dim, latent_dim)
		                                )
		self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim),  ACT(),
		                             nn.Linear(hidden_dim, input_dim),
		                             )
		self.mean_net = nn.Sequential(nn.Linear(latent_dim, hidden_dim), ACT(),
		                              nn.Linear(hidden_dim, 1),
		                              )
		self.log_var_net = nn.Sequential(nn.Linear(n_domains, hidden_dim), ACT(),
		                                 nn.Linear(hidden_dim, latent_dim))
		self.domain_embedding = nn.Embedding(n_domains, n_domains)
		self.adj_mat = ADJMatrix(latent_dim)
		self.input_dim = input_dim
		# self.prior_mean_net.apply(weights_init('xavier'))
		self.register_buffer('sure_mask', torch.zeros([latent_dim, latent_dim]))
		self.register_buffer('sure_adj', torch.zeros([latent_dim, latent_dim]))

	def get_adj(self, soft=False, current=False):
		adj = self.adj_mat(soft)
		device = adj.device
		final_adj = self.sure_mask.to(device) * self.sure_adj.to(device) + (1 - self.sure_mask.to(device)) * adj
		if current:
			final_adj = adj[~self.sure_mask.to(torch.bool).to(device)]
			cur_size = np.sqrt(final_adj.size(0)).astype(int)
			if cur_size == 0:
				return torch.zeros([1, 1])
			final_adj = final_adj.view(cur_size, -1)
			assert final_adj.shape[0] == final_adj.shape[1]  # square
		return final_adj

	@torch.no_grad()
	def find_sinknodes_and_fix(self, epoch):
		current_adj = self.get_adj(current=True)
		if len(current_adj) == 0:
			return False
		adj = self.get_adj()
		for i in range(current_adj.size()[0]):
			if current_adj[:, i].sum() == 0:
				self.sure_mask[:, i] = 1
				self.sure_mask[i, :] = 1
				self.sure_adj[:, i] = adj[:, i]
				self.sure_adj[i, :] = adj[i, :]
				print('>>>>>>>>>>. sink node found at %d <<<<<<<<<<<<<' % i)
			# print(self.get_adj(current=True).size(), ' >>>>> current sizees ')

		# when we are sure of all the nodes or only one node
		nonzero_col = 0
		for i in range(self.latent_dim):
			if torch.sum(self.sure_mask[:, i]) != self.sure_mask.shape[1]:
				nonzero_col += 1
		should_stop = (nonzero_col == 0)
		if torch.sum(current_adj) == 0:
			should_stop = False
		return should_stop

	def forward(self, x, d):
		domain_embedding = self.domain_embedding(d.argmax(dim=1).squeeze())
		xd = torch.cat([x, d], dim=1)
		xd = x
		mu = self.encoder_mu(xd)
		logvar = self.encoder_logvar(xd)
		z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

		all_z = z
		xhat = self.decoder(all_z)
		prior_log_var = self.log_var_net(domain_embedding)

		adj_mat = self.get_adj().to(x.device)

		z_rep = z.repeat_interleave(self.latent_dim, dim=0).view(len(x), self.latent_dim, self.latent_dim)
		y_rep = d.repeat_interleave(self.latent_dim, 0)
		adj_mat_rep = adj_mat.unsqueeze(0).repeat(len(x), 1, 1)
		assert z_rep.shape == adj_mat_rep.shape == (len(x), self.latent_dim, self.latent_dim)
		mask_z_rep = adj_mat_rep * z_rep
		mask_z_rep = mask_z_rep.view(len(x) * self.latent_dim, self.latent_dim)
		cond_mean = self.mean_net(mask_z_rep)
		cond_mean = cond_mean.view(len(x), self.latent_dim)

		current_adj_mat = self.get_adj(current=True)
		return {'xhat': xhat, 'mu': mu, 'logvar': logvar, 'adj_mat': adj_mat, 'prior_log_var': prior_log_var,
		        'z': z, 'cond_mean': cond_mean, 'current_adj_mat': current_adj_mat,
		        }
