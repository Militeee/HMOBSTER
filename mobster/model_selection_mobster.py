from mobster.likelihood_calculation import *
import numpy as np


def ICL(lk,data, params, tail, params_noccf):
    bic = BIC(lk, data, params_noccf)
    entropy = compute_entropy(params, tail)
    return bic + entropy

def BIC(lk, data, params_noccf):
    n_params = calculate_number_of_params(params_noccf)
    n = number_of_samples(data)
    return np.log(n) * n_params - 2 * lk

def AIC(lk, params_noccf):
    n_params = calculate_number_of_params(params_noccf)
    return 2 * n_params - 2 * lk

def likelihood(lk):
    res = 0
    for k in lk.keys():
        res += log_sum_exp(lk[k]).sum()
    return res


def calculate_number_of_params(params):

    res = 0
    for i in params:
        if type(params[i]) is dict:
            next
        else:
            res += np.prod(params[i].shape)
    return res


def number_of_samples(data):

    res = 0
    for k in data:
        res += np.prod(data[k].shape)
    return res