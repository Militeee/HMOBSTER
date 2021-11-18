from pandas.core.common import flatten
from tqdm import trange
import pyro
import torch
import numpy as np


theo_clonal_list = {
    "1:0" : 1,
    "1:1" : 1,
    "2:0" : 2,
    "2:1" : 2,
    "2:2" : 2
}

theo_allele_list = {
    "1:0" : 1,
    "1:1" : 2,
    "2:0" : 2,
    "2:1" : 3,
    "2:2" : 4
}

theo_clonal_means_list = {
    "1:0" : torch.tensor(0.9999),
    "1:1" : torch.tensor(0.5),
    "2:0" : torch.tensor([0.5,0.9999]),
    "2:1" : torch.tensor([1/3.,2/3.]),
    "2:2" : torch.tensor([0.25,0.5])
}

ccf_adjust =  {
    "1:0" : torch.tensor(1),
    "1:1" : torch.tensor(0.5),
    "2:0" : torch.tensor(0.5),
    "2:1" : torch.tensor(1/3.),
    "2:2" : torch.tensor(0.25)
}


def flatten_list(l):
    return list(flatten(l))


def log_sum_exp(args):
    c = torch.amax(args, dim=0)
    return c + torch.log(torch.sum(torch.exp(args - c), axis=0))


def get_theo_clones(data):
    karyos = list(data.keys())
    theoretical_num_clones = [theo_clonal_list[kr] for kr in karyos]
    return theoretical_num_clones

def get_clones_counts(theoretical_num_clones):
    uniques = list(set(theoretical_num_clones))
    uniques = {k : 0 for k in uniques}
    res = [None] * len(theoretical_num_clones)
    for i in range(len(theoretical_num_clones)):
        res[i] = uniques[theoretical_num_clones[i]]
        uniques[theoretical_num_clones[i]] += 1
    return(res)


def compute_entropy(params, tail):

    res = 0
    posts = params["cluster_probs"]
    for k in posts:
        posts_k = posts[k]
        if tail:
            posts_k = posts_k[1:]
        log_post_k = torch.log(posts_k + 0.000001)
        post_k_entr = posts_k * log_post_k
        post_k_entr = torch.sum(post_k_entr, axis = 0)
        post_k_entr = -1 * torch.sum(post_k_entr)
        res += post_k_entr
    return res



def format_parameters_for_export(data, params, tail, K):
    res = {k : 0 for k in data.keys()}
    theoretical_num_clones = get_theo_clones(data)
    clones_count = get_clones_counts(theoretical_num_clones)
    for i, k in enumerate(res):
        res[k] = format_parameters_for_export_aux(data, params,k, i, theoretical_num_clones, clones_count, tail, K)
    return res


def format_parameters_for_export_aux(data, params,k, i, theo_clones, counts_clone, tail, K):

    j = counts_clone[i]
    if theo_clones[i] == 2:
        beta_concentration1 = params['a_2'][:,j] * params['avg_number_of_trials_beta'][i]
        beta_concentration2 = (1 - params['a_2'][:,j]) * params['avg_number_of_trials_beta'][i]
    else:
        beta_concentration1 = params['a_1'][:, j] * params['avg_number_of_trials_beta'][i]
        beta_concentration2 = (1 - params['a_1'][:, j]) * params['avg_number_of_trials_beta'][i]

    mixture_weights = params['param_weights_{}'.format(theo_clones[i])][j, :].detach().numpy()
    if tail == 1:
        tail_weights = params['param_tail_weights'][i, :].detach().numpy()
        mixture_weights = mixture_weights * tail_weights[1]
        mixture_weights = np.insert(mixture_weights, 0, tail_weights[0])

    NV = data[k][:,0]
    DP = data[k][:,1]
    VAF = NV / DP

    ca, order_vec = rename_clusters(np.argmax(params["cluster_probs"][k].detach().numpy(), axis = 0), tail, theo_clones[i], K)

    res = {"cluster_probs" : params["cluster_probs"][k].detach().numpy(),
           "cluster_assignments" : ca,
           "cluster_types" : order_vec,
           "mixture_probs" : mixture_weights,
          "beta_concentration1" : beta_concentration1.detach().numpy(),
          "beta_concentration2" : beta_concentration2.detach().numpy(),
           "ccf_subclones" : params["ccf_priors"].detach().numpy(),
            "dispersion_noise" : 1/ params['prc_number_of_trials_beta']
        }
    if tail == 1:
        res["tail_shape"] = params['tail_mean'].detach().numpy()
        res["tail_scale"] = np.min(VAF.detach().numpy())
        res["tail_noise"] =  1/params['alpha_noise'][i].detach().numpy()

    return res


def rename_clusters(x,tail, theo_c, K):

    res = np.empty(len(x), dtype='object')
    base_idx = 0
    clonal_num = 1
    subclonal_num = 1
    order_vec = []

    ###  IDENTIFY TAIL MUTATIONS ###
    if tail == 1:
        res[x == 0] = "Tail"
        order_vec.append("Tail")
        base_idx += 1

    ### IDENTIFY CLONAL MUTATIONS ###
    for i in range(base_idx, base_idx + theo_c):
        res[x == i] = "C" + str(clonal_num)
        order_vec.append("C" + str(clonal_num))
        clonal_num += 1

    ### IDENTIFY SUBCLONAL MUTATIONS ###
    for i in range(base_idx + theo_c, base_idx + theo_c + K):
        res[x == i] = "S" + str(subclonal_num)
        order_vec.append("S" + str(subclonal_num))
        subclonal_num += 1

    return res,np.array(order_vec)

def include_ccf(data, params, K):

    if K == 0:
        return params
    kar = list(data.keys())
    cccfs_2 = [ccf_adjust[k] * params["purity"] for k in kar if theo_clonal_list[k] == 2]
    cccfs_1 = [ccf_adjust[k] * params["purity"] for k in kar if theo_clonal_list[k] == 1]

    correct_ccfs2 = torch.tensor(cccfs_2)
    correct_ccfs1 = torch.tensor(cccfs_1)

    ccf_cat2 = torch.outer(params["ccf_priors"], correct_ccfs2)
    ccf_cat1 = torch.outer(params["ccf_priors"], correct_ccfs1)

    if "a_2" in params:
        params['a_2'] = torch.cat([params['a_2'], ccf_cat2.reshape([K,-1]) ], 0)
    if "a_1" in params:
        params['a_1'] = torch.cat([params['a_1'], ccf_cat1.reshape([K,-1]) ], 0)

    return params

try:
    from typing import Iterable
except ImportError:
    from collections import Iterable




def flatten(items):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

def retrieve_params():
    param_names = pyro.get_param_store()
    res = {nms: pyro.param(nms) for nms in param_names}
    return res

def collect_weights(pars):
    pars = list(flatten([value.detach().tolist() for key, value in pars.items() if key.find("weights") >= 0]))
    return(np.array(pars))

def collect_params(pars):
    pars = list(flatten([value.detach().tolist() for key, value in pars.items()]))
    return(np.array(pars))

def collect_params_no_noise(pars):
    beta = list(flatten([value.detach().tolist() for key, value in pars.items() if key.find("a_") >= 0]))
    overdisp = pars["avg_number_of_trials_beta"]
    mix = list(flatten([value.detach().tolist() for key, value in pars.items() if key.find("weights") >= 0]))
    ret = list(flatten([beta,overdisp ,mix]))

    if "u" in pars.keys():
        tail_mean = pars["u"].detach().tolist()
        ret = list(flatten([ret, tail_mean]))

    if "ccf_priors" in pars.keys():
        subclones = pars["ccf_priors"].detach().tolist()
        ret = list(flatten([ret, subclones]))
    return(np.array(ret))


