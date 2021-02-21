from mobster.likelihood_calculation import *


def retrieve_posterior_probs(data, parameters, tail):
    lks = compute_likelihood_from_params(data, parameters, tail, tsum = False)
    res = {k : 0 for k in data.keys()}
    for k in res:
        lks_k = lks[k]
        norm_fact = log_sum_exp(lks_k)
        res[k] = torch.exp(lks_k - norm_fact)
    parameters["cluster_probs"] = res
    return parameters
