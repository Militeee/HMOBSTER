from pandas.core.common import flatten
from tqdm import trange
import pyro
import torch
import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

try:
    from typing import Iterable
except ImportError:
    from collections import Iterable


def theo_clonal_num(kr, range = True):
    kr = kr.split(":")
    kr = [int(k) for k in kr]
    if range:
        return torch.arange(1, max(kr) + 1)
    return max(kr)

def theo_clonal_tot(kr):
    kr = kr.split(":")
    kr = [int(k) for k in kr]
    return sum(kr)


def flatten_list(l):
    return list(flatten(l))


def log_sum_exp(args):
    c = torch.amax(args, dim=0)
    return c + torch.log(torch.sum(torch.exp(args - c), axis=0))


def get_theo_clones(data):
    karyos = list(data.keys())
    theoretical_num_clones = [theo_clonal_num(kr, range = False) for kr in karyos]
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



def format_parameters_for_export(data, params, tail, K, purity, truncated_pareto, subclonal_prior, multi_tails):
    res = {k : 0 for k in data.keys()}
    theoretical_num_clones = get_theo_clones(data)
    clones_count = get_clones_counts(theoretical_num_clones)
    for i, k in enumerate(res):
        res[k] = format_parameters_for_export_aux(data, params,k, i, theoretical_num_clones, clones_count, tail, K, purity, truncated_pareto, subclonal_prior, multi_tails)
    return res


def format_parameters_for_export_aux(data, params,k, i, theo_clones, counts_clone, tail, K, purity, truncated_pareto, subclonal_prior, multi_tails):

    beta_concentration1 = params['a_{}'.format(i)] * params['avg_number_of_trials_beta'][i]
    beta_concentration2 = (1 - params['a_{}'.format(i)]) * params['avg_number_of_trials_beta'][i]


    mixture_weights = params['param_weights_{}'.format(i)].detach().numpy()

    if K > 0:
        ccfs_torch = (params["ccf_priors"] * purity) / (2 * (1 - purity) + theo_clonal_tot(k) * purity)
        ccfs = [ccfs_torch.detach().tolist()]

    if tail == 1:
        tail_weights = params['param_tail_weights'][i, :].detach().numpy()
        mixture_weights = mixture_weights * tail_weights[1]
        mixture_weights = np.insert(mixture_weights, 0, tail_weights[0])

        if theo_clones[i] > 1:
            if truncated_pareto:
                b_max = torch.amin(params['a_{}'.format(i)])
                if K > 0 and multi_tails:
                    bm = [b_max.detach().tolist()]
                    b_max = torch.Tensor(list(flatten([bm, ccfs])))

            else:
                b_max = torch.tensor(0.999)
        else:
            if truncated_pareto:
                b_max = params['a_{}'.format(i)][0]
                if K > 0 and multi_tails:
                    b_max -=  torch.amax(ccfs_torch).item()
                    bm = [b_max.detach().tolist()]
                    b_max = torch.Tensor(list(flatten([bm, ccfs])))
            else:
                b_max = torch.tensor(0.999)

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
            "dispersion_noise" : 1/ params['prc_number_of_trials_beta'][i].detach().numpy()
        }

    if K > 0:

        res["ccf_subclones"] = params["ccf_priors"].detach().numpy()
        res["loc_subclones"] = ccfs_torch.detach().numpy()
        if subclonal_prior == "Moyal":
            res["scale_subclonal"] = (1./params["scale_subclonal_{}".format(i)]).reshape([K]).detach().numpy()
        else:
            res["n_trials_subclonal"] = params["n_trials_subclonal_{}".format(i)].detach().numpy()

    if tail == 1:
        res["tail_shape"] = np.exp(params['tail_mean'].detach().numpy())
        res["tail_scale"] = scale_pareto(VAF).detach().numpy()
        res["tail_noise"] =  (1/params['alpha_noise']).detach().numpy()
        res["tail_higher"] = b_max.detach().numpy()
        if K > 0 and truncated_pareto and multi_tails:
            res["multi_tail_weights"] = params['multitail_weights'][i].detach().numpy()
        else:
            res["multi_tail_weights"] = np.array(1)
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



def flatten(items):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

def retrieve_params(CUDA = False):
    param_names = pyro.get_param_store()
    if CUDA:
        res = {nms: pyro.param(nms).cpu() for nms in param_names}
    else:
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

def scale_pareto(VAF, max_vaf = 0.1):
    NBINS = 100
    hist = torch.histc(VAF, NBINS, 0, 1)
    vals = torch.cumsum(1/NBINS * torch.ones(NBINS),0)

    idx = torch.where(hist[0:-1] > hist[1:])

    best_scale = vals[idx[0][0]]
    
    if best_scale.detach().item() > max_vaf:
        return torch.min(VAF) - 1e-10
    else:
        return best_scale - 1e-10
    
def initialize_subclone(VAF, karyo, purity, K_number, tail, subclones):
    
    km = KMeans(K_number).fit(VAF.numpy().reshape(-1, 1))
    
    sorted_centroids = np.sort(np.array(km.cluster_centers_.flatten()))
    
       
    idx_left = tail
    idx_right = idx_left + subclones
     
    subclonal_centroid =  torch.tensor(sorted_centroids[idx_left:idx_right])
    
    
    
    subclonal_centroid = (subclonal_centroid  * (purity * karyo + (1 - purity) * 2)) / purity
    
    if(torch.sum(subclonal_centroid > 0.9)) > 0: 
        subclonal_centroid[subclonal_centroid > 0.9] = 1 / torch.arange(start=2, end = torch.sum(subclonal_centroid > 0.9) + 2)
        
    
    return subclonal_centroid

def to_cpu(tns):
    if type(tns) is dict:
        (tn.to("cpu") for tn in tns.values())
    else:
        tns.cpu()

