from mobster.utils_mobster import *
import torch
import pyro.distributions as dist
from mobster.BoundedPareto import *
from mobster.Moyal import *
from mobster.ParetoBinomial import *


def beta_lk(beta_a, beta_b, K, NV, DP):
    lk = torch.ones(K, len(NV))
    if K == 1:
        return dist.BetaBinomial(beta_a, beta_b, total_count=DP).log_prob(NV)
    for k in range(K):
        lk[k, :] = dist.BetaBinomial(beta_a[k], beta_b[k], total_count=DP).log_prob(NV)
    return lk


def moyal_lk(p, K, NV, DP):
    lk = torch.ones(K, len(NV))
    if K == 1:
        return dist.Binomial(probs=p, total_count=DP).log_prob(NV)
    for k in range(K):
        lk[k, :] = dist.Binomial(probs=p[k], total_count=DP).log_prob(NV)
    return lk


def pareto_lk(p, NV, DP, K, weights):
    # a = p * (rho / (1 - rho))
    # b = (1 / rho) - a - 1
    if weights is not None:
        lk = torch.ones(K + 1, len(NV))
        for k in range(K + 1):
            lk[k, :] = torch.log(weights[k]) + dist.Binomial(probs=p[k], total_count=DP).log_prob(NV)
        return log_sum_exp(lk)
    else:
        return dist.Binomial(probs=p, total_count=DP).log_prob(NV)

def pareto_binomial_lk(alpha, U, L, NV, DP, K, weights):

    if weights is not None:
        lk = torch.ones(K + 1, len(NV))
        for k in range(K + 1):
            lk[k, :] = torch.log(weights[k]) + ParetoBinomial(alpha, U[k], L,trials=DP).log_prob(NV)
        return log_sum_exp(lk)
    else:
        return ParetoBinomial(alpha, U, L,trials=DP).log_prob(NV)


def final_lk(pareto, beta, weights):
    if len(beta.shape) == 1:
        dim0, dim1 = 1, beta.shape[0]
    else:
        dim0, dim1 = beta.shape[0], beta.shape[1]

    lk = torch.ones(1 + dim0, dim1)
    lk[0, :] = torch.log(weights[0]) + pareto
    lk[1:, :] = torch.log(weights[1]) + beta

    return lk


def compute_likelihood_from_params(data, params, tail, truncated_pareto, purity, K, subclonal_prior, multi_tails, min_vaf_scale_tail, tsum=True):
    theoretical_num_clones = get_theo_clones(data)
    clones_count = get_clones_counts(theoretical_num_clones)

    if tsum:
        lk = 0
    else:
        lk = [None] * len(theoretical_num_clones)

    for i, k in enumerate(data):

        tmp = compute_likelihood_from_params_aux(data[k], tail, truncated_pareto,
                                                 params, i,
                                                 theoretical_num_clones, clones_count,
                                                 k, purity, K, subclonal_prior, multi_tails, min_vaf_scale_tail)
        if tsum:
            tmp = log_sum_exp(tmp)
            lk += torch.sum(tmp)
        else:
            lk[i] = tmp

    if not tsum:
        ks = data.keys()
        lk = {k: v for k, v in zip(ks, lk)}
    return lk


def calculate_lk_multitail_params_old(NV, DP, lower, tail, b_max, weights, K, truncated_pareto, multi_tails):

    LINSPACE = 1000
    x = torch.linspace(lower.item(), torch.max(b_max).item(), LINSPACE)
    if K == 0 or not truncated_pareto or not multi_tails:
        y_1 = BoundedPareto(lower, tail, b_max).log_prob(
            x).exp()
        y_2 = dist.Binomial(probs=x.repeat([NV.shape[0], 1]).reshape([LINSPACE, -1]), total_count=DP).log_prob(
            NV).exp()
        return torch.trapz(y_1.reshape([LINSPACE, 1]) * y_2, x=x, dim=0).log()

    lk = torch.zeros(K + 1, len(NV))
    for i in range(K + 1):
        y_1 = BoundedPareto(lower, tail, b_max[i]).log_prob(
            x).exp()
        y_2 = dist.Binomial(probs=x.repeat([NV.shape[0], 1]).reshape([LINSPACE, -1]), total_count=DP).log_prob(
            NV).exp()

        lk[i, :] = torch.trapz(y_1.reshape([LINSPACE, 1]) * y_2, x=x, dim=0).log() + torch.log(weights[i])
    return (log_sum_exp(lk))

def calculate_lk_multitail_params(NV, DP, lower, tail, b_max, weights, K, truncated_pareto, multi_tails):

    if K == 0 or not truncated_pareto or not multi_tails:
        return ParetoBinomial(tail, b_max, lower, trials=DP).log_prob(NV)

    lk = torch.zeros(K + 1, len(NV))
    for i in range(K + 1):

        lk[i, :] = ParetoBinomial(tail, b_max[i], lower, trials=DP).log_prob(NV) + torch.log(weights[i])
    return (log_sum_exp(lk))


def calculate_lk_moyal_params(NV, DP, lower, loc, scale, b_max, K):
    LINSPACE = 1000
    x = torch.linspace(lower, torch.max(b_max).item(), LINSPACE)
    if K > 1:
        lk = torch.ones(K, len(NV))
    b_max = torch.max(b_max)
    if K == 1:
        y_1 = BoundedMoyal(loc, scale, lower, b_max).log_prob(
            x).exp()
        y_2 = dist.Binomial(probs=x.repeat([NV.shape[0], 1]).reshape([LINSPACE, -1]), total_count=DP).log_prob(
            NV).exp()
        return torch.trapz(y_1.reshape([LINSPACE, 1]) * y_2, x=x, dim=0).log()
    for i in range(K):
        y_1 = BoundedMoyal(loc[i], scale[i], lower, b_max).log_prob(
            x).exp()
        y_2 = dist.Binomial(probs=x.repeat([NV.shape[0], 1]).reshape([LINSPACE, -1]), total_count=DP).log_prob(
            NV).exp()
        lk[i, :] = torch.trapz(y_1.reshape([LINSPACE, 1]) * y_2, x=x, dim=0).log()
    return lk


def compute_likelihood_from_params_aux(data, tail, truncated_pareto, params, i, theo_clones, counts_clone, karyo,
                                       purity, K, subclonal_prior, multi_tails, min_vaf_scale_tail):
    NV = data[:, 0]
    DP = data[:, 1]
    VAF = NV / DP
    b_max = 0

    beta = 0
    pareto = 0



    theo_peaks = (theo_clonal_num(karyo) * purity - 1e-9) / (2 * (1 - purity) + theo_clonal_tot(karyo) * purity)

    if K > 0:
        ccfs = (params["ccf_priors"] * purity) / (2 * (1-purity) + theo_clonal_tot(karyo) * purity)


    beta = beta_lk(params['a_{}'.format(i)] * params['avg_number_of_trials_beta'][i],
                   (1 - params['a_{}'.format(i)]) * params['avg_number_of_trials_beta'][i],
                   len(params['a_{}'.format(i)]),
                   NV, DP)
    b_max = torch.amin(theo_peaks)
    if tail:
        if truncated_pareto:
            b_max_tail = b_max
            weights = torch.tensor(1)
            if K > 0 and multi_tails:
                b_max_tail -= torch.max(ccfs)
                b_max_tail = torch.Tensor(list(flatten([[b_max_tail.detach().tolist()], ccfs.detach().tolist()])))
                b_max_tail[b_max_tail < (torch.min(VAF) - 1e-5)] = torch.min(VAF)
                weights = params["multitail_weights"][i, :]
        else:
            weights = torch.tensor(1)
            b_max_tail = torch.tensor(0.999)

        pareto = calculate_lk_multitail_params(NV, DP, scale_pareto(VAF, min_vaf_scale_tail),
                                               torch.exp(params['tail_mean']), b_max_tail,
                                               weights, K, truncated_pareto, multi_tails)

    if K > 0:
        if subclonal_prior == "Moyal":
            scale = params["scale_subclonal_{}".format(i)]
            subclonal_lk = calculate_lk_moyal_params(NV, DP, torch.min(VAF) - 1e-5,
                                                     ccfs, 1/scale, b_max,
                                                     K)
        else:
            n_trials = params["n_trials_subclonal_{}".format(i)]
            subclonal_lk = beta_lk(ccfs * n_trials,
                                   (1 - ccfs) * n_trials,
                                   K,
                                   NV, DP)


    if tail and (K > 0):

        not_neutral = torch.vstack([beta, subclonal_lk]) + \
                      torch.log(params['param_weights_{}'.format(i)]).reshape(
                          [K + theo_clones[i], -1])
        lk = final_lk(pareto, not_neutral, params['param_tail_weights'][i, :])
    if tail and not (K > 0):
        not_neutral = beta + torch.log(params['param_weights_{}'.format(i)]).reshape(
                          [theo_clones[i], -1])
        lk = final_lk(pareto, not_neutral, params['param_tail_weights'][i, :])
    if not tail and (K > 0):
        lk = torch.vstack([beta, subclonal_lk]) + torch.log(
            params['param_weights_{}'.format(i)].reshape(
                [K + theo_clones[i], -1]))
    if not tail and not (K > 0):
        lk = beta + torch.log(params['param_weights_{}'.format(i)]).reshape(
                          [theo_clones[i], -1])

    return lk
