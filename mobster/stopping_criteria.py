import numpy as np
from mobster.utils_mobster import *


def all_stopping_criteria(old, new, e):
    old = collect_params(old)
    new = collect_params(new)
    diff_mix = np.abs(old - new) / np.abs(old)
    if np.all(diff_mix < e):
        return True
    return False

def mixture_stopping_criteria(old, new, e):
    old = collect_weights(old)
    new = collect_weights(new)
    diff_mix = np.abs(old - new) / np.abs(old)
    if np.all(diff_mix < e):
        return True
    return False

def all_no_noise_stopping_criteria(old, new, e):
    old = collect_params_no_noise(old)
    new = collect_params_no_noise(new)
    diff_mix = np.abs(old - new)/ np.abs(old)
    if np.all(diff_mix < e):
        return True
    return False