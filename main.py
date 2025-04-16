import sys

import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
import os
from data import create_data
from trainer import Trainer
from tqdm import tqdm
from utils import plot, compute_mcc, get_latest_run_id, collect_first_domain
import torchvision
import torch.nn as nn
import torch.nn.functional as F

argparse = argparse.ArgumentParser()
argparse.add_argument('--case', type=str, default='chain', choices=['chain', 'fork'])
argparse.add_argument('--result_dir', type=str, default='runs')
argparse.add_argument('--n_domains', type=int, default=50)
argparse.add_argument('--latent_dim', type=int, default=None)
argparse.add_argument('--batch_size', type=int, default=8192)
argparse.add_argument('--n_samples', type=int, default=10000)
argparse.add_argument('--lambda_rec', type=float, default=100)
argparse.add_argument('--lambda_kl', type=float, default=10)
argparse.add_argument('--lambda_sparse', type=float, default=0.0)
argparse.add_argument('--check_epoch', type=int, default=20)
argparse.add_argument('--seed', type=int, default=1)
args = argparse.parse_args()


if args.case == 'chain':
	args.lambda_kl = 1.0
	args.lambda_rec = 10.
	args.seed = 123
	args.lambda_sparse = 0.01
	args.check_epoch = 5
	ground_truth = torch.tensor([[0,0,0],
	                             [1,0,0],
	                             [0,1,0],])


elif args.case == 'fork':
	args.lambda_kl = 1.0
	args.lambda_rec = 10.
	args.seed = 1
	args.lambda_sparse = 0.01
	args.check_epoch = 5
	ground_truth = torch.tensor([[0,0,0],
	                             [0,0,0],
	                             [1,1,0],])


print('ground truth', ground_truth)

run_dir = os.path.join(args.result_dir, get_latest_run_id(args.result_dir)+'-case%s-rec%s-kl%s-sparse%s-truth-%s'%(args.case, args.lambda_rec, args.lambda_kl, args.lambda_sparse, str(args.latent_dim)))
os.makedirs(run_dir, exist_ok=True)
device = 'cpu'
torch.manual_seed(args.seed)
np.random.seed(args.seed)

file = 'example_%s.npz' % args.case
if os.path.exists(file):
	all_data, all_latents, domain_labels = np.load(file)['all_data'], np.load(file)['all_latents'], np.load(file)['domain_labels']
	all_data = torch.from_numpy(all_data).float()
	all_latents = torch.from_numpy(all_latents).float()
	domain_labels = torch.from_numpy(domain_labels).long()
	print('loading data from %s' % file)
else:
	all_data, all_latents, domain_labels = create_data(case=args.case, n_samples=args.n_samples, n_domains=args.n_domains)



first_domain_data, first_domain_latents, first_domain_labels = collect_first_domain(all_data, all_latents, domain_labels)
dataset = torch.utils.data.TensorDataset(all_data, all_latents, domain_labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
first_domain_dataset = torch.utils.data.TensorDataset(first_domain_data, first_domain_latents, first_domain_labels)
first_domain_dataloader = torch.utils.data.DataLoader(first_domain_dataset, batch_size=args.batch_size, shuffle=True)
#dataloader, loader_test = create_mnist()
trainer = Trainer(input_dim=all_data.shape[-1], n_domains=args.n_domains, device=device, latent_dim=args.latent_dim,
                  args=args,
                  ).to(device)
fix_x, fix_z, fix_d = next(iter(dataloader_test))
fix_x = fix_x[:64].to(device)
fix_d = fix_d[:64].to(device)

total_iter = 0
should_stop = False
recon_losses = []
kl_losses = []
steps = []
for epoch in range(1, 150+1):
	pbar = tqdm(dataloader)
	for x, z, d in pbar:
		total_iter += 1
		loss_dict = trainer.step(x.to(device), d.to(device), epoch, z.to(device))
		pbar.set_description('Epoch: %d, KL: %.4f, rec: %.4f sparse: %.4f' %(epoch, loss_dict['kl'], loss_dict['rec'],
		                                                                                  loss_dict['loss_sparse'],
		                                                                     ))
		recon_losses.append(loss_dict['rec'])
		kl_losses.append(loss_dict['kl'])
		steps.append(total_iter)
	if epoch % args.check_epoch == 0:
		should_stop = trainer.model.find_sinknodes_and_fix(epoch)
		
	if epoch % 1 == 0:
		print(trainer.model.get_adj(soft=True))
	trainer.decay_lr()
	zs, true_zs = trainer.evaluate(dataloader)
	zs = zs.cpu().detach().numpy()
	true_zs = true_zs.cpu().detach().numpy()
	plot(true_zs, zs, fname=os.path.join(run_dir, 'scatter_case%s_epoch%d.jpg'%(args.case, epoch)))
	first_domain_zs, first_domain_true_zs = trainer.evaluate(first_domain_dataloader)
	print(len(true_zs), len(zs), ' >>>>>>>>>>>>>>>>>>>> ', len(first_domain_zs), len(first_domain_true_zs))
	plot(first_domain_true_zs, first_domain_zs, fname=os.path.join(run_dir, 'scatter_first_domain_case%s_epoch%d.jpg'%(args.case, epoch)))
	metric_dict = compute_mcc(true_zs, zs)
	print('lr:%.7f' % trainer.optim.param_groups[0]['lr'])
	print(trainer.model.get_adj(soft=True))
	#print(loss_dict['ele_kl'])
	if should_stop:
		break



