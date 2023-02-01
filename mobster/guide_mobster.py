import pyro as pyro
import numpy as np
from pyro.infer import MCMC, NUTS, Predictive, HMC, config_enumerate
from pyro.infer.autoguide import AutoDelta, init_to_sample
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from torch.distributions import constraints
import mobster.utils_mobster as mut
from mobster.likelihood_calculation import *



def guide(data, K=1, tail=1, truncated_pareto = True,subclonal_prior = "Moyal",multi_tail = False,  purity=0.96, clonal_beta_var=1., number_of_trials_clonal_mean=100.,
          number_of_trials_subclonal=300., number_of_trials_k=300., prior_lims_clonal=[1., 10000.],alpha_precision_concentration = 100, alpha_precision_rate=0.1,
          prior_lims_k=[1., 10000.], epsilon_ccf = 0.01, max_min_subclonal_ccf = [0.05,0.95], k_means_init = True, min_vaf_scale_tail = 0.01):


    karyos = list(data.keys())

    # Here we calculate the theoretical number of clonal clusters
    theoretical_num_clones = [mut.theo_clonal_num(kr, range=False) for kr in karyos]

    # Calculate the theoretical clonal means, wihch can be analytically defined for simple karyotypes, and then multiply them by the ploidy
    theoretical_clonal_means = [mut.theo_clonal_num(kr) for kr in karyos]

    theo_allele_list = [mut.theo_clonal_tot(kr) for kr in karyos]


    counts_clones = dict()
    for i in theoretical_num_clones:
        counts_clones[i] = counts_clones.get(i, 0) + 1

    
    if truncated_pareto and K > 0 and multi_tail:
        multitail_weights = pyro.param("multitail_weights", 1 / (K + 1) * torch.ones([len(karyos), K + 1]),  constraint=constraints.simplex)

    a_prior = pyro.param("tail_mean", torch.zeros(1) + 0.1, constraint=constraints.real)

    alpha_precision_par = pyro.param("alpha_noise",
                                     dist.Gamma(concentration=alpha_precision_concentration,
                                                rate=alpha_precision_rate).mean,
                                     constraint=constraints.positive)




    avg_number_of_trials_beta = pyro.param("avg_number_of_trials_beta", torch.ones(len(karyos)) * number_of_trials_clonal_mean, constraint=constraints.positive)
    precision_number_of_trials_beta = pyro.param("prc_number_of_trials_beta", torch.ones(len(karyos)) * 20, constraint=constraints.positive)


    alpha_prior = pyro.sample('u', dist.Delta(a_prior))
    
    first_kar = list(data.keys())[0]
    
    VAFS_init = data[first_kar]
    VAFS_init = VAFS_init[:,0] / VAFS_init[:,1]
    
    if k_means_init and K > 0:
        ccf_init, _ = initialize_subclone(VAFS_init, mut.theo_clonal_tot(first_kar), purity, K + tail + mut.theo_clonal_num(first_kar,range = False), tail, K).sort()
    else:
        ccf_init = ((torch.tensor(1) - 0.001) / (K + 1)) * torch.arange(1,K+1)

    ccf_priors = pyro.param("ccf_priors",ccf_init,constraint=constraints.unit_interval)
    
    if K != 0:
        with pyro.plate("subclones", K):
            subclonal_ccf = pyro.sample("sb_ccf", dist.Delta(ccf_priors))


    if tail == 1:
        weights_tail = pyro.param("param_tail_weights",  (torch.tensor([1/(K+2), (1 + K)/(K + 2)]).repeat([len(karyos), 1]) ), constraint=constraints.simplex)


    for kr in pyro.plate("kr", len(karyos)):


        NV = data[karyos[kr]][:, 0]
        DP = data[karyos[kr]][:, 1]
        VAF = NV / DP

        prior_overdispersion = pyro.sample('prior_overdisp_{}'.format(kr),
                                           dist.Delta(avg_number_of_trials_beta[kr]))
        prec_overdispersion = pyro.sample('prec_overdisp_{}'.format(kr),
                                          dist.Delta(precision_number_of_trials_beta[kr]))

        weights_param = pyro.param("param_weights_{}".format(kr), (1 / (K + theoretical_num_clones[kr])) * torch.ones(K + theoretical_num_clones[kr]),
                                   constraint=constraints.simplex)

        pyro.sample('weights_{}'.format(kr), dist.Delta(weights_param).to_event(1))

        theo_peaks = (theoretical_clonal_means[kr] * purity - 1e-9) / (2 * (1 - purity) + theo_allele_list[kr] * purity)

        # Mean parameter
        a_theo = theo_peaks



        a = pyro.param('a_{}'.format(kr),
                         a_theo.reshape([theoretical_num_clones[kr], 1]),
                         constraint=constraints.unit_interval)


        with pyro.plate("clones_{}".format(kr)):
            pyro.sample('beta_clone_mean_{}'.format(kr), dist.Delta(a).to_event(1))
            pyro.sample('beta_clone_n_samples_{}'.format(kr), dist.LogNormal(torch.log(prior_overdispersion), 1/prec_overdispersion))

        if K > 0:
            with pyro.plate("subclones_{}".format(kr)):

                k_means = pyro.sample('beta_subclone_mean_{}'.format(kr),
                                      dist.Uniform(((subclonal_ccf - epsilon_ccf) * purity) / (2 * (1-purity) + theo_allele_list[kr] * purity),
                                                   ((subclonal_ccf + epsilon_ccf)  * purity)/ (2 * (1-purity) + theo_allele_list[kr] * purity)))

                if subclonal_prior == "Moyal":
                    scale_subclonal_param = pyro.param("scale_subclonal_{}".format(kr), torch.tensor(100.) * torch.ones(K),
                                                       constraint=constraints.real)
                    scale_subclonal = pyro.sample("scale_moyal_{}".format(kr), dist.Delta(scale_subclonal_param))
                    pyro.sample("subclones_prior_{}".format(kr),
                                                BoundedMoyal(k_means - 1./scale_subclonal * (EULER_MASCHERONI + torch.log(torch.tensor(2))) , 1./scale_subclonal, torch.min(torch.amin(VAF), torch.tensor(min_vaf_scale_tail)) - 1e-5,
                                                             torch.amin(theo_peaks)).to_event(1))
                    ba = BoundedMoyal(k_means, torch.exp(scale_subclonal), torch.min(torch.amin(VAF), torch.tensor(min_vaf_scale_tail)) - 1e-5,
                                                             torch.amin(theo_peaks))

                    #ba = pyro.sample("subclones_prior_{}".format(kr), Moyal(k_means, torch.exp(scale_subclonal)))


                else:
                    n_trials_subclonal = pyro.param("n_trials_subclonal_{}".format(kr), torch.ones(K) * number_of_trials_subclonal,
                                                    constraint=constraints.positive)
                    num_trials_subclonal = pyro.sample("N_subclones_{}".format(kr), dist.Delta(n_trials_subclonal))
                    pyro.sample("subclones_prior_{}".format(kr),
                                                dist.Beta(k_means * num_trials_subclonal,
                                                          (1 - k_means) * num_trials_subclonal))


        if tail == 1:
            # K = K + tail
            pyro.sample('weights_tail_{}'.format(kr), dist.Delta(weights_tail[kr]).to_event(1))

            alpha_precision = pyro.sample('alpha_precision_{}'.format(kr), dist.Delta(alpha_precision_par))
            #alpha_prior = torch.clamp(alpha_prior, -100,100)
            pyro.sample("alpha_noise_{}".format(kr),
                                dist.LogNormal(alpha_prior,
                                               1 / alpha_precision))



