# import numpy as np
def ELBO_stopping_criteria(old, new, e=0.01):
    diff_ELBO = old - new
    cutoff = e * old
    if diff_ELBO > cutoff:
        return True
    return False

