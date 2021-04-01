from pandas.core.common import flatten
from tqdm import trange
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
    "1:0" : torch.tensor(0.95),
    "1:1" : torch.tensor(0.5),
    "2:0" : torch.tensor([0.5,0.95]),
    "2:1" : torch.tensor([0.333,0.666]),
    "2:2" : torch.tensor([0.25,0.5])
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



def format_parameters_for_export(data, params, tail):
    res = {k : 0 for k in data.keys()}
    theoretical_num_clones = get_theo_clones(data)
    clones_count = get_clones_counts(theoretical_num_clones)
    for i, k in enumerate(res):
        res[k] = format_parameters_for_export_aux(data, params,k, i, theoretical_num_clones, clones_count, tail)
    return res


def format_parameters_for_export_aux(data, params,k, i, theo_clones, counts_clone, tail):

    j = counts_clone[i]
    if theo_clones[i] == 2:
        beta_concentration1 = params['a_2'][:,j] * params['b_2'][:,j]
        beta_concentration2 = (1 - params['a_2'][:,j]) * params['b_2'][:, j]
    else:
        beta_concentration1 = params['a_1'][:, j] * params['b_1'][:, j]
        beta_concentration2 = (1 - params['a_1'][:, j]) * params['b_1'][:, j]

    mixture_weights = params['param_weights_{}'.format(theo_clones[i])][j, :].detach().numpy()
    if tail == 1:
        tail_weights = params['param_tail_weights'][i, :].detach().numpy()
        mixture_weights = mixture_weights * tail_weights[1]
        mixture_weights = np.insert(mixture_weights, 0, tail_weights[0])


    ca, order_vec = rename_clusters(np.argmax(params["cluster_probs"][k].detach().numpy(), axis = 0), tail, theo_clones[i])

    res = {"cluster_probs" : params["cluster_probs"][k].detach().numpy(),
           "cluster_assignments" : ca,
           "cluster_types" : order_vec,
           "mixture_probs" : mixture_weights,
          "beta_concentration1" : beta_concentration1.detach().numpy(),
          "beta_concentration2" : beta_concentration2.detach().numpy()
        }
    if tail == 1:
        res["tail_shape"] = params['tail_mean'].detach().numpy()
        res["tail_scale"] = np.min(data[k].detach().numpy())
        res["tail_noise"] =  1/params['alpha_noise'][i].detach().numpy()

    return res


def rename_clusters(x,tail, theo_c):
    res = np.empty(len(x), dtype='object')
    uniq_clusts = np.unique(x)
    base_idx = 0
    clonal_num = 1
    subclonal_num = 1
    order_vec = np.array([],dtype='object' )
    ###  IDENTIFY TAIL MUTATIONS ###
    if tail == 1:
        res[x == 0] = "Tail"
        np.append(order_vec,"Tail")
        base_idx += 1

    ### IDENTIFY CLONAL MUTATIONS ###
    for i in range(base_idx, base_idx + theo_c):
        res[x == i] = "C" + str(clonal_num)
        np.append(order_vec,"C" + str(clonal_num))
        clonal_num += 1

    ### IDENTIFY SUBCLONAL MUTATIONS ###
    for i in range(base_idx + theo_c,len(uniq_clusts)):
        res[x == i] = "S" + str(subclonal_num)
        np.append(order_vec, "S" + str(subclonal_num))
        subclonal_num += 1

    return res,order_vec

