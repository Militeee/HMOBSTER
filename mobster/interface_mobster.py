import pandas as pd
import numpy as np
import seaborn as sns

import pyro as pyro
import numpy as np
from pyro.infer.autoguide import AutoDelta
import pyro.poutine as poutine
import torch
import mobster
from mobster.stopping_criteria import *


def fit_mobster(data, K, tail=1, purity=0.96, alpha_prior_sd=0.3, number_of_trials_clonal_mean=500.,number_of_trials_k=300.,
         prior_lims_clonal=[0.1, 100000.], prior_lims_k=[0.1, 100000.], stopping = ELBO_stopping_criteria, lr = 0.05, max_it = 5000):

    model = mobster.model
    guide = mobster.guide

    svi = svi = pyro.infer.SVI(model=model,
                     guide=guide,
                     optim=pyro.optim.Adam({"lr": lr}),
                     loss=pyro.infer.TraceGraph_ELBO())

    print('Running MOBSTER on {} karyotypes'.format(len(data)), flush=True)
    params = {
        'K' : K,
        'tail' : tail,
        'purity' : purity,
        'alpha_prior_sd' : alpha_prior_sd,
        'number_of_trials_clonal_mean' : number_of_trials_clonal_mean,
        'number_of_trials_k' : number_of_trials_k,
        'prior_lims_clonal' : prior_lims_clonal,
        'prior_lims_k' : prior_lims_k,
    }
    run(data, params, svi, stopping, max_it)

    params_dict = retrieve_params()

    params_dict["assignements_probs"] = mobster.retrieve_posterior_probs(data, params_dict)

    ICL = mobster.ICL(data, params_dict)



def run(data, params, svi, stopping, max_it):

    data_dict = params
    data_dict["data"] = data
    pyro.clear_param_store()
    old = 10^10
    new = svi.step(**data_dict)
    losses = []
    for i in range(1,max_it) :
        #if stopping(old, new):
        #    continue
        loss = svi.step(**data_dict)
        #old = new
        #new = loss
        losses.append(loss)
        print(loss)


def retrieve_params():
    param_names = pyro.get_param_store()
    res = {nms: pyro.param(nms) for nms in param_names}
    return res