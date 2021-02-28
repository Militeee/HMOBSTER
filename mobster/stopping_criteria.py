import numpy as np
def ELBO_stopping_criteria(old, new, e=0.01):
    diff_ELBO = np.abs(old - new)
    cutoff = e * np.abs(old)
    if diff_ELBO < cutoff:
        return True
    return False

