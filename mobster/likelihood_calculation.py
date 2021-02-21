from mobster.utils_mobster import *

def beta_lk(beta_a, beta_b, weights, K, data):
    lk = torch.ones(K, len(data))
    if K == 1:
        return torch.log(weights) + dist.Beta(beta_a, beta_b).log_prob(data)
    for k in range(K):
        lk[k, :] = torch.log(weights[k]) + dist.Beta(beta_a[k], beta_b[k]).log_prob(data)
    return lk

def final_lk(pareto, beta, weights):
    lk = torch.ones(1 + beta.shape[0], beta.shape[1])
    lk[0, :] = torch.log(weights[0]) + pareto
    lk[1:(1 + beta.shape[1]), :] = torch.log(weights[1]) + beta
    print(lk.shape)
    return lk

def compute_likelihood_from_params(data, params, tail, tsum = True):

    theoretical_num_clones = get_theo_clones(data)
    clones_count = get_clones_counts(theoretical_num_clones)
    if tsum:
        lk = 0
    else :
        lk = [None] * len(theoretical_num_clones)
    for i,k in enumerate(data):

        if tail:
            tmp = compute_likelihood_from_params_tail(data[k], params, i, theoretical_num_clones, clones_count)
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



def compute_likelihood_from_params_tail(data, params, i, theo_clones, counts_clone):
    j = counts_clone[i]
    if theo_clones[i] == 2:
        beta = beta_lk(params['a_2'][:,j] * params['b_2'][:,j],
                       (1 - params['a_2'][:,j]) * params['b_2'][:, j],
                       params['param_weights_{}'.format(theo_clones[i])][j, :],
                       len(params['a_2'][:, j]),
                       data)
        pareto = dist.Pareto(torch.min(data) - 1e-5, params['ap']).log_prob(data)
        lk = final_lk(pareto, beta, params['param_tail_weights'][i, :])
    else:
        beta = beta_lk(params['a_1'][:, j] * params['b_1'][:, j],
                       (1 - params['a_1'][:, j]) * params['b_1'][:, j],
                       params['param_weights_{}'.format(theo_clones[i])][j,:],
                       len(params['a_1'][:, j]),
                       data)
        pareto = dist.Pareto(torch.min(data) - 1e-5, params['ap']).log_prob(data)
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
