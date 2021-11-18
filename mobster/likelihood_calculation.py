from mobster.utils_mobster import *
import torch
import pyro.distributions as dist
from mobster.BoundedPareto import *


def beta_lk(beta_a, beta_b, weights, K, NV, DP):
    lk = torch.ones(K, len(NV))
    if K == 1:
        return torch.log(weights) + dist.BetaBinomial(beta_a, beta_b, total_count=DP).log_prob(NV)
    for k in range(K):
        lk[k, :] = torch.log(weights[k]) + dist.BetaBinomial(beta_a[k], beta_b[k], total_count=DP).log_prob(NV)
    return lk


def pareto_lk(p, rho, NV, DP):
    # a = p * (rho / (1 - rho))
    # b = (1 / rho) - a - 1
    return dist.Binomial(probs=p, total_count=DP).log_prob(NV)


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
            tmp = compute_likelihood_from_params_tail(data[k],truncated_pareto,
                                                      params, i,
                                                      theoretical_num_clones, clones_count,
                                                      k)
            if tsum:
                tmp = log_sum_exp(tmp)
                lk += torch.sum(tmp)
            else:
                lk[i] = tmp

        else:
            tmp = compute_likelihood_from_params_no_tail(data[k], params,
                                                         i, theoretical_num_clones,
                                                         clones_count)
            if tsum:
                tmp = log_sum_exp(tmp)
                lk += torch.sum(tmp)
            else:
                lk[i] = tmp

    if not tsum:
        ks = data.keys()
        lk = {k:v for k,v in zip(ks, lk)}
    return lk



def compute_likelihood_from_params_tail(data, truncated_pareto, params, i, theo_clones, counts_clone, karyo):

    j = counts_clone[i]
    NV = data[:, 0]
    DP = data[:, 1]
    VAF = NV/DP
    b_max = 0
    LINSPACE = 2000

    if theo_clones[i] == 2:

        b_max = torch.amin(params['a_2'][1:2,j])

        if truncated_pareto:
            x = torch.linspace(torch.min(VAF),b_max.item(), LINSPACE)
            y_1 = BoundedPareto(torch.min(VAF) - 1e-5, params['tail_mean'] * theo_allele_list[karyo] , b_max).log_prob(x).exp()
        else:
            x = torch.linspace(torch.min(VAF), 0.999, LINSPACE)
            y_1 = BoundedPareto(torch.min(VAF) - 1e-5, params['tail_mean'] * theo_allele_list[karyo] , 1).log_prob(x).exp()
        y_2 = dist.Binomial(probs = x.repeat([NV.shape[0],1]).reshape([LINSPACE,-1]), total_count=DP).log_prob(NV).exp()


        pareto = torch.trapz(y_1.reshape([LINSPACE, 1]) * y_2, x =  x, dim = 0).log()

        beta = beta_lk(params['a_2'][:,j] * theo_allele_list[karyo],
                       (1 - params['a_2'][:,j]) * theo_allele_list[karyo],
                       params['param_weights_{}'.format(theo_clones[i])][j, :],
                       len(params['a_2'][:, j]),
                       NV, DP)
        lk = final_lk(pareto, beta, params['param_tail_weights'][i, :])
    else:

        if params['a_1'][:,j].shape[0] > 1:
            b_max = params['a_1'][1, j]
        else:
            b_max = params['a_1'][:,j]

        if truncated_pareto:
            x = torch.linspace(torch.min(VAF), b_max.item(), LINSPACE)
            y_1 = BoundedPareto(torch.min(VAF) - 1e-5,
                                params['tail_mean'] * theo_allele_list[karyo],
                                b_max).log_prob(x).exp()

        else:
            x = torch.linspace(torch.min(VAF), 0.999, LINSPACE)
            y_1 = BoundedPareto(torch.min(VAF) - 1e-5,
                              params['tail_mean'] * theo_allele_list[karyo],
                                                 1).log_prob(x).exp()
        y_2 = dist.Binomial(probs = x.repeat([NV.shape[0], 1]).reshape([LINSPACE, -1]), total_count=DP).log_prob(NV).exp()

        pareto = torch.trapz(y_1.reshape([LINSPACE, 1]) * y_2, x= x, dim=0).log()

        beta = beta_lk(params['a_1'][:, j] * params['avg_number_of_trials_beta'][i],
                       (1 - params['a_1'][:, j]) * params['avg_number_of_trials_beta'][i],
                       params['param_weights_{}'.format(theo_clones[i])][j,:],
                       len(params['a_1'][:, j]),
                       NV, DP)

        lk = final_lk(pareto, beta, params['param_tail_weights'][i, :])
    return lk

def compute_likelihood_from_params_no_tail(data, params, i, theo_clones, counts_clone):



    NV = data[:,0]
    DP = data[:,1]
    j = counts_clone[i]
    if theo_clones[i] == 2:
        lk = beta_lk(params['a_2'][:, j] * params['avg_number_of_trials_beta'][i],
                       (1 - params['a_2'][:, j]) * params['avg_number_of_trials_beta'][i],
                       params['param_weights_{}'.format(theo_clones[i])][j, :],
                       len(params['a_2'][:, j]),
                       NV, DP)
    else:
        lk = beta_lk(params['a_1'][:, j] * params['avg_number_of_trials_beta'][i],
                       (1 - params['a_1'][:, j]) * params['avg_number_of_trials_beta'][i],
                       params['param_weights_{}'.format(theo_clones[i])][j, :],
                       len(params['a_1'][:, j]),
                       NV, DP)
    return lk
