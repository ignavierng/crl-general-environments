import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.utils
from scipy.stats import ortho_group
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from utils import plot
from torchvision.datasets import MNIST
from torchvision import transforms
def create_data(case='chain', n_samples=10000, n_domains=100):
    print('>>>>>>>> Creating data for case %s <<<<<<<<<<<'%case)
    if case=='chain':
        return generate_data_chain(n_samples=n_samples, n_domains=n_domains)
    elif case == 'fork':
        return generate_data_fork(n_samples=n_samples, n_domains=n_domains)

def sigmoid(x):
    return 1 / (1 + np.exp(-x)
                )
def leaky_relu(x, alpha=0.2):
    return np.where(x > 0, x, alpha * x)



def transform_data(x, dim=4):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    input_x = x
    scale = np.sqrt(x.shape[1])
    scale= 1
    mat1 = (np.random.uniform(-2, 2., [x.shape[1], dim]))
    x = x@mat1
    x = leaky_relu(x)
    mat2 = (np.random.uniform(-2, 2., [dim, 1]))
    x = (x@mat2)
    out = x.reshape(-1)
    assert out.shape[0] == x.shape[0]
    return out/np.sqrt(input_x.shape[-1])


def generate_data_chain(n_samples=10000, n_domains=100, n_layers=2, dim=3):
    eps1 = np.random.normal(0, 1, n_samples)
    eps2 = np.random.normal(0, 1, n_samples)
    eps3 = np.random.normal(0, 1, n_samples)
    inv_mats = [ortho_group.rvs(dim) for _ in range(n_layers)]
    scale = np.random.uniform(0.5, 5, [n_domains, dim, dim])
    self_scale = np.random.uniform(0.1, 5, [n_domains, dim])
    bias = np.random.uniform(-1, 1, [n_domains, dim])
    #bias = np.zeros_like(self_scale)

    ground_truth_latents = []
    domain_labels = []
    data = []
    for i in range(n_domains):
        z1 = eps1 * self_scale[i, 0]
        z2 = eps2 * self_scale[i, 1] + transform_data(z1)
        z3 = eps3 * self_scale[i, 2] + transform_data(z2)
        ground_truth = np.array([[0,1,0],
                                 [0,0,1],
                                 [0,0,0]])

        z = np.array([z1, z2, z3]).reshape(-1, 3)
        x = z
        x = x@inv_mats[0]
        for j in range(1, n_layers):
            x = np.where(x>0, x, x*0.1)
            x = x@inv_mats[j]
        x = torch.from_numpy(x).float()
        data.append(x)
        ground_truth_latents.append(torch.from_numpy(z).float())
        domain_labels += ([i]*n_samples)
    all_data = torch.cat(data)
    all_latents = torch.cat(ground_truth_latents)
    domain_labels = torch.tensor(domain_labels)
    return all_data, all_latents, domain_labels

def generate_data_fork(n_samples=10000, n_domains=100, n_layers=2, dim=3):
    eps1 = np.random.normal(0, 1, n_samples)
    eps2 = np.random.normal(0, 1, n_samples)
    eps3 = np.random.normal(0, 1, n_samples)
    inv_mats = [ortho_group.rvs(dim) for _ in range(n_layers)]
    scale = np.random.uniform(0.5, 5, [n_domains, dim, dim])
    self_scale = np.random.uniform(0.1, 5, [n_domains, dim])
    bias = np.random.uniform(-1, 1, [n_domains, dim])
    ground_truth_latents = []
    domain_labels = []
    data = []
    mus = []
    tmp_zero = np.zeros([n_samples]).reshape(-1)
    for i in range(n_domains):
        z1 = eps1 * self_scale[i, 0]
        z2 = (eps2 * self_scale[i, 1])
        add_z3 = transform_data(np.concatenate([z1.reshape(-1,1), z2.reshape(-1,1)], axis=1))
        z3 = (eps3 * self_scale[i, 2] + add_z3)/4

        print(z1.mean(), z2.mean(), z3.mean(), ' >>>>>> sca;esss')
        ground_truth = np.array([[0,0,0],
                                 [0,0,0],
                                 [1,1,0]
                                 ])
        z = np.stack([z1, z2, z3], 1)
        assert z.shape == (n_samples, dim)
        x = deepcopy(z)
        x = x@inv_mats[0]
        for j in range(1, n_layers):
            x = np.where(x>0, x, x*0.2)
            x = x@inv_mats[j]
        x = torch.from_numpy(x).float()
        data.append(x)
        ground_truth_latents.append(torch.from_numpy(z).float())
        domain_labels += ([i]*n_samples)
    randinx = np.random.permutation(n_samples*n_domains)
    all_data = torch.cat(data)[randinx]
    all_latents = torch.cat(ground_truth_latents)[randinx]
    domain_labels = torch.tensor(domain_labels)[randinx]
    return all_data, all_latents, domain_labels


