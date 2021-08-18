from mobster.utils_mobster import *
import torch
import pyro.distributions as dist
from mobster.BoundedPareto import *


def beta_lk(beta_a, beta_b, weights, K, data):
    lk = torch.ones(K, len(data))
    if K == 1:
        return torch.log(weights) + dist.Beta(beta_a, beta_b).log_prob(data)
    for k in range(K):
        lk[k, :] = torch.log(weights[k]) + dist.Beta(beta_a[k], beta_b[k]).log_prob(data)
    return lk

def final_lk(pareto, beta, weights):
    if len(beta.shape) == 1:
        dim0, dim1 = 1,beta.shape[0]
    else:
        dim0, dim1 = beta.shape[0], beta.shape[1]
    lk = torch.ones(1 + dim0, dim1)
    lk[0, :] = torch.log(weights[0]) + pareto
    lk[1:(1 + dim1), :] = torch.log(weights[1]) + beta
    return lk

def compute_likelihood_from_params(data, params, tail, truncated_pareto, tsum = True):

    theoretical_num_clones = get_theo_clones(data)
    clones_count = get_clones_counts(theoretical_num_clones)

    if tsum:
        lk = 0
    else :
        lk = [None] * len(theoretical_num_clones)

    for i,k in enumerate(data):

        if tail == 1:
            tmp = compute_likelihood_from_params_tail(data[k],truncated_pareto, params, i, theoretical_num_clones, clones_count)
            if tsum:
                tmp = log_sum_exp(tmp)
                lk += torch.sum(tmp)
            else:
                lk[i] = tmp

        else:
            tmp = compute_likelihood_from_params_no_tail(data[k], params, i, theoretical_num_clones, clones_count)
            if tsum:
                tmp = log_sum_exp(tmp)
                lk += torch.sum(tmp)
            else:
                lk[i] = tmp

    if not tsum:
        ks = data.keys()
        lk = {k:v for k,v in zip(ks, lk)}
    return lk



def compute_likelihood_from_params_tail(data, truncated_pareto, params, i, theo_clones, counts_clone):
    j = counts_clone[i]
    b_max = 0
    if theo_clones[i] == 2:
        b_max = torch.amin(params['a_2'][1:2,j])
        beta = beta_lk(params['a_2'][:,j] * params['b_2'][:,j],
                       (1 - params['a_2'][:,j]) * params['b_2'][:, j],
                       params['param_weights_{}'.format(theo_clones[i])][j, :],
                       len(params['a_2'][:, j]),
                       data)
        if truncated_pareto:
            pareto = BoundedPareto(torch.min(data) - 1e-5, params['tail_mean'], b_max).log_prob(data)
        else:
            pareto = dist.Pareto(torch.min(data) - 1e-5, params['tail_mean']).log_prob(data)
        lk = final_lk(pareto, beta, params['param_tail_weights'][i, :])
    else:
        if params['a_1'][:,j].shape[0] > 1:
            b_max = params['a_1'][1, j]
        else:
            b_max = params['a_1'][:,j]
        beta = beta_lk(params['a_1'][:, j] * params['b_1'][:, j],
                       (1 - params['a_1'][:, j]) * params['b_1'][:, j],
                       params['param_weights_{}'.format(theo_clones[i])][j,:],
                       len(params['a_1'][:, j]),
                       data)
        if truncated_pareto:
            pareto = BoundedPareto(torch.min(data) - 1e-5, params['tail_mean'], b_max).log_prob(data)
        else:
            pareto = dist.Pareto(torch.min(data) - 1e-5, params['tail_mean']).log_prob(data)
        lk = final_lk(pareto, beta, params['param_tail_weights'][i, :])
    return lk

def compute_likelihood_from_params_no_tail(data, params, i, theo_clones, counts_clone):
    j = counts_clone[i]
    if theo_clones[i] == 2:
        lk = beta_lk(params['a_2'][:, j] * params['b_2'][:, j],
                       (1 - params['a_2'][:, j]) * params['b_2'][:, j],
                       params['param_weights_{}'.format(theo_clones[i])][j, :],
                       len(params['a_2'][:, j]),
                       data)
    else:
        lk = beta_lk(params['a_1'][:, j] * params['b_1'][:, j],
                       (1 - params['a_1'][:, j]) * params['b_1'][:, j],
                       params['param_weights_{}'.format(theo_clones[i])][j, :],
                       len(params['a_1'][:, j]),
                       data)
    return lk
