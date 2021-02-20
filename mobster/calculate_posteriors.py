from likelihood_calculation import*


def retrieve_posterior_probs(data, parameters, tail):
    lk, norm_factors = compute_likelihood_from_params(data, parameters, tail)
    res = {k : 0 for k in data.keys()}
    for k in res:
        pass


