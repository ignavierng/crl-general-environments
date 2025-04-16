import matplotlib.pyplot as plt
import numpy as np
import os

import torch


def compute_mcc(z1, z2):
    Ncomp = z1.shape[-1]
    from scipy.stats import spearmanr
    from scipy.optimize import linear_sum_assignment

    CorMat = (np.abs(np.corrcoef(z1.T, z2.T)))[:Ncomp, Ncomp:]
    ii = linear_sum_assignment(-1 * CorMat)
    mcc_pearson = CorMat[ii].mean()
    print(CorMat[ii])

    rho, _ = np.abs(spearmanr(z1, z2))
    CorMat_s = rho[:Ncomp, Ncomp:]
    ii_s = linear_sum_assignment(-1 * CorMat_s)
    mcc_spearman = CorMat_s[ii_s].mean()
    print(CorMat_s[ii_s])

    print('MCC_pearson:', mcc_pearson)
    print('MCC_spearman:', mcc_spearman)

    metric_dict = {}
    pearson_vec = CorMat[ii]
    for i in range(len(pearson_vec)):
        metric_dict['Pearson/MCC_%d' % (i+1)] = pearson_vec[i]
    metric_dict['Pearson/MCC_avg'] = mcc_pearson
    spearman_vec = CorMat_s[ii_s]
    for i in range(len(spearman_vec)):
        metric_dict['Spearman/MCC_%d' % (i+1)] = spearman_vec[i]
    metric_dict['SPearman/MCC_avg'] = mcc_spearman
    return metric_dict

def plot(data, estimated, fname):
    m = data.shape[1]
    n = estimated.shape[1]
    fig, axes = plt.subplots(m, n, figsize=(n*5, m*5))
    print(m, n, ' >>>>>>>>>>>> MNNNN')
    # Add titles and labels for subplots
    for j in range(m):
        for i in range(n):
            axes[j, i].scatter(data[:, j], estimated[:, i], alpha=0.5)
            #if i == n - 1:
            #    axes[j, i].set_xlabel(f"True {j + 1}", fontdict={'fontsize': 20})
            #if j == 0:
            #    axes[j, i].set_ylabel(f"Est {i + 1}", fontdict={'fontsize': 20})
            axes[j, i].set_title(f"True {j + 1} vs Est {i + 1}", fontdict={'fontsize': 20})

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def get_latest_run_id(directory_path):
    """
    Reads the specified directory and returns the latest run_id based on subfolders named with a
    seven-digit run_id at the beginning ('%07d...').

    Args:
    directory_path (str): The path to the directory to scan.

    Returns:
    str: The latest run_id as a string formatted as '%07d', or None if no valid run_id found.
    """
    latest_id = -1  # Start with an impossible low value
    latest_run_id = None

    # List all items in the directory
    for item in os.listdir(directory_path):
        # Ensure the item is a directory and starts with at least 7 digits
        if os.path.isdir(os.path.join(directory_path, item)) and item[:7].isdigit():
            # Convert the first 7 characters to an integer
            current_id = int(item[:7])
            # Update latest_id and latest_run_id if this ID is the largest found so far
            if current_id > latest_id:
                latest_id = current_id
                latest_run_id = f"{current_id+1:07d}"  # Format the ID back to a string with leading zeros
    if latest_run_id is None:
        return '%07d' % 0
    return latest_run_id


def collect_first_domain(x, latents,  labels):
    new_x = []
    new_latents = []
    new_labels = []
    for i in range(len(labels)):
        if labels[i] == 1:
            new_x.append(x[i])
            new_labels.append(labels[i])
            new_latents.append(latents[i])
    new_x = np.stack(new_x)
    new_labels = np.array(new_labels)
    new_latents = np.stack(new_latents)
    new_x = torch.from_numpy(new_x).float()
    new_labels = torch.from_numpy(new_labels)
    new_latents = torch.from_numpy(new_latents)
    return new_x, new_latents,new_labels