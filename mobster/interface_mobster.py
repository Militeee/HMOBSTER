import pandas as pd
import numpy as np
import seaborn as sns
import tqdm

import pyro as pyro
import numpy as np
import math
from pyro.infer.autoguide import AutoDelta
import pyro.poutine as poutine
import torch
import mobster
from mobster.utils_mobster import *
from mobster.stopping_criteria import *
import mobster.model_selection_mobster as ms
from mobster.calculate_posteriors import *
from pyro.util import ignore_jit_warnings


def fit_mobster(data, K, tail=1, truncated_pareto = True, purity=0.96, number_of_trials_clonal_mean=500.,number_of_trials_k=300.,
                alpha_precision_concentration = 5, alpha_precision_rate=0.1,
         prior_lims_clonal=[0.1, 100000.], prior_lims_k=[0.1, 100000.], stopping = ELBO_stopping_criteria, lr = 0.05,
                max_it = 5000, e = 0.001, compile = False, CUDA = False, seed = 3, lrd_gamma = 0.1):


    pyro.set_rng_seed(seed)

    if CUDA:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if compile:
        loss = pyro.infer.JitTraceGraph_ELBO
    else:
        loss = pyro.infer.TraceGraph_ELBO

    model = mobster.model
    guide = mobster.guide

    lrd = lrd_gamma ** (1/max_it)

    svi = pyro.infer.SVI(model=model,
                     guide=guide,
                     optim=pyro.optim.ClippedAdam({"lr": lr, "lrd" : lrd}),
                     loss= loss())

    print('Running MOBSTER on {} karyotypes with {} subclones.'.format(len(data), K), flush=True)
    if tail == 1:
        print("Fitting a model with tail", flush = True)
    else:
        print("Fitting a model without tail", flush=True)
    params = {
        'K' : K,
        'tail' : tail,
        'truncated_pareto' : truncated_pareto,
        'purity' : purity,
        'alpha_precision_concentration' : alpha_precision_concentration,
        'alpha_precision_rate' : alpha_precision_rate,
        'number_of_trials_clonal_mean' : number_of_trials_clonal_mean,
        'number_of_trials_k' : number_of_trials_k,
        'prior_lims_clonal' : prior_lims_clonal,
        'prior_lims_k' : prior_lims_k
    }
    loss = run(data, params, svi, stopping, max_it, e)

    params_dict_noccf = retrieve_params()
    params_dict = include_ccf(data, params_dict_noccf, K)

    print("", flush=True)
    print("Computing cluster assignements.", flush=True)
    params_dict = retrieve_posterior_probs(data,truncated_pareto,  params_dict, tail)


    ### Caclculate information criteria
    print("Computing information criteria.", flush=True)
    likelihood = ms.likelihood(data, params_dict, tail, truncated_pareto)
    AIC = ms.AIC(data, params_dict, tail, truncated_pareto,params_dict_noccf)
    BIC = ms.BIC(data, params_dict, tail, truncated_pareto,params_dict_noccf)
    ICL = ms.ICL(data, params_dict, tail, truncated_pareto, params_dict_noccf)

    params_dict = format_parameters_for_export(data, params_dict, tail,K)



    information_dict =  {"likelihood": likelihood.detach().numpy(),
                         "AIC": AIC.detach().numpy(),
                        "BIC" : BIC.detach().numpy(),
                        "ICL" : ICL.detach().numpy()}


    final_dict = {
        "information_criteria" : information_dict,
        "model_parameters" : params_dict,
        "run_parameters" : params,
        "loss": np.array(loss)
    }
    print("Done!", flush = True)

    return final_dict



def run(data, params, svi, stopping, max_it, e):

    N = ms.number_of_samples(data)
    data_dict = params.copy()
    data_dict["data"] = data
    pyro.clear_param_store()
    loss = new = svi.step(**data_dict)
    losses = []
    t = trange(max_it, desc='Bar desc', leave=True)
    for i in t:

        t.set_description('ELBO: {:.9f}  '.format(loss / N))
        t.refresh()

        loss = svi.step(**data_dict)
        losses.append(loss)

        old, new = new, loss

        if stopping(old, new, e):
            break
        if np.isinf(loss) or math.isinf(loss):
            break

    return losses

def retrieve_params():
    param_names = pyro.get_param_store()
    res = {nms: pyro.param(nms) for nms in param_names}
    return res