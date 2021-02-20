from pandas.core.common import flatten
import pyro.distributions as dist
import torch
import numpy as np

def flatten_list(l):
    return list(flatten(l))


def log_sum_exp(args):
    c = torch.amax(args, dim=0)
    return c + torch.log(torch.sum(torch.exp(args - c), axis=0))


def get_theo_clones(data):
    karyos = list(data.keys())
    major = [int(str(i).split(":")[0]) for i in karyos]
    minor = [int(str(i).split(":")[1]) for i in karyos]
    theoretical_num_clones = [1 if (mn == 0 or mn == mj) else 2 for mj, mn in zip(major, minor)]
    return theoretical_num_clones

def get_clones_counts(theoretical_num_clones):
    uniques = list(set(theoretical_num_clones))
    uniques = {k : 0 for k in uniques}
    res = [None] * len(theoretical_num_clones)
    for i in range(len(theoretical_num_clones)):
        res[i] = uniques[theoretical_num_clones[i]]
        uniques[theoretical_num_clones[i]] += 1
    return(res)


def compute_entropy():
    pass

def compute_number_of_params():
    pass