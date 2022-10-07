from mobster.likelihood_calculation import *


def retrieve_posterior_probs(data, truncated_pareto, parameters, tail, purity, K, subclonal_prior, multi_tails, min_vaf_scale_tail):
    lks = compute_likelihood_from_params(data, parameters, tail, truncated_pareto,purity, K, subclonal_prior, multi_tails,min_vaf_scale_tail,  tsum = False)
    res = {k : 0 for k in data.keys()}
    for k in res:
        lks_k = lks[k]
        if len(lks_k.shape) == 1:
            res[k] = torch.ones([1,lks_k.shape[0]])
        else:
            norm_fact = log_sum_exp(lks_k)
            res[k] = torch.exp(lks_k - norm_fact)
    parameters["cluster_probs"] = res
    return parameters, lks


