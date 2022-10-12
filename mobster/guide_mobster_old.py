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


def guide_old(data, K=1, tail=1, truncated_pareto = True,subclonal_prior = "Moyal",multi_tail = False,  purity=0.96, clonal_beta_var=1., number_of_trials_clonal_mean=100.,
          number_of_trials_subclonal=300., number_of_trials_k=300., prior_lims_clonal=[1., 10000.],alpha_precision_concentration = 100, alpha_precision_rate=0.1,
          prior_lims_k=[1., 10000.], epsilon_ccf = 0.002, max_min_subclonal_ccf = [0.05,0.95], k_means_init = True):


    karyos = list(data.keys())



    # Here we calculate the theoretical number of clonal clusters
    theoretical_num_clones = [mut.theo_clonal_list[kr] for kr in karyos]

    # Calculate the theoretical clonal means, wihch can be analytically defined for simple karyotypes, and then multiply them by the ploidy
    theoretical_clonal_means = [mut.theo_clonal_means_list[kr] for kr in karyos]

    theo_allele_list = [mut.theo_allele_list[kr] for kr in karyos]

    theo_peaks = [(mut.theo_clonal_means_list[kr] * purity) / (2 * (1 - purity) + mut.theo_allele_list[kr] * purity) for kr in karyos]

    counts_clones = dict()
    for i in theoretical_num_clones:
        counts_clones[i] = counts_clones.get(i, 0) + 1

    index_2 = [i for i, j in enumerate(theoretical_num_clones) if j == 2]
    index_1 = [i for i, j in enumerate(theoretical_num_clones) if j == 1]
    if truncated_pareto and K > 0:
        multitail_weights = pyro.param("multitail_weights", 1 / (K + 1) * torch.ones([len(karyos), K + 1]),  constraint=constraints.simplex)

    a_prior = pyro.param("tail_mean", torch.ones(1), constraint=constraints.positive)

    alpha_precision_par = pyro.param("alpha_noise",
                                     dist.Gamma(concentration=alpha_precision_concentration,
                                                rate=alpha_precision_rate).mean * torch.ones([len(karyos)]),
                                     constraint=constraints.positive)



    avg_number_of_trials_beta = pyro.param("avg_number_of_trials_beta", torch.ones(len(karyos)) * number_of_trials_clonal_mean, constraint=constraints.positive)
    precision_number_of_trials_beta = pyro.param("prc_number_of_trials_beta", torch.ones(len(karyos)) * 20, constraint=constraints.positive)


    alpha_prior = pyro.sample('u', dist.Delta(a_prior))
    
    VAFS_by_karyo = { k:(v[:,0] / v[:,1]) for k,v in data.items() }

    if k_means_init and K :
        ccf_init = torch.hstack([ initialize_subclone(v, theo_allele_list[k], purity, K + tail, tail, K) for v,k in VAFS_by_karyo.items()])
    else:
        ccf_init = ((torch.min(torch.tensor(1)) - 0.001) / (K + 1)) * torch.arange(1,K+1)

    ccf_priors = pyro.param("ccf_priors",
        ccf_init, constraint=constraints.unit_interval)
    
    if K != 0:
        with pyro.plate("subclones", K):
            subclonal_ccf = pyro.sample("sb_ccf", dist.Delta(ccf_priors).to_event(0))

    idx1 = 0
    idx2 = 0

    if tail == 1:
        weights_tail = pyro.param("param_tail_weights",  (torch.tensor([1/(K+2), (1 + K)/(K + 2)]).repeat([len(karyos), 1]) ), constraint=constraints.simplex)
    if 2 in theoretical_num_clones:
        weights_param_2 = pyro.param("param_weights_2", (1 / (K + 2)) * torch.ones([len(index_2), K + 2]),
                                     constraint=constraints.simplex)
    if 1 in theoretical_num_clones:
        weights_param_1 = pyro.param("param_weights_1", (1 / (K + 1)) * torch.ones([len(index_1), K + 1]),
                                     constraint=constraints.simplex)



    for kr in pyro.plate("kr", len(karyos)):

        NV = data[karyos[kr]][:, 0]
        DP = data[karyos[kr]][:, 1]
        VAF = NV / DP

        prior_overdispersion = pyro.sample('prior_overdisp_{}'.format(kr),
                                           dist.Delta(avg_number_of_trials_beta[kr]))
        prec_overdispersion = pyro.sample('prec_overdisp_{}'.format(kr),
                                          dist.Delta(precision_number_of_trials_beta[kr]))

        if theoretical_num_clones[kr] == 2:
            pyro.sample('weights_{}'.format(kr), dist.Delta(weights_param_2[idx2]).to_event(1))

            # Mean parameter when the number of clonal picks is 2
            a_2_theo = torch.cat([theo_peaks[i] for i in index_2]).reshape([counts_clones[2], 2])

            a_2_theo = torch.transpose(a_2_theo,0,1)

            a21 = pyro.param('a_2',
                             a_2_theo.reshape([2, len(index_2)]),
                             constraint=constraints.unit_interval)


            with pyro.plate("clones_{}".format(kr), 2):
                pyro.sample('beta_clone_mean_{}'.format(kr), dist.Delta(a21[:, idx2]))
                pyro.sample('beta_clone_n_samples_{}'.format(kr), dist.LogNormal(torch.log(prior_overdispersion), 1/prec_overdispersion))
            idx2 += 1


        else:

            pyro.sample('weights_{}'.format(kr), dist.Delta(weights_param_1[idx1]).to_event(1))

            # Mean parameter when the number of clonal picks is 1
            a_1_theo = torch.tensor([theo_peaks[i] for i in index_1]).reshape([1, counts_clones[1]])

            a11 = pyro.param('a_1',
                             a_1_theo.reshape([1 , len(index_1)]),
                             constraint=constraints.unit_interval)

            with pyro.plate("clones_{}".format(kr), 1):
                pyro.sample('beta_clone_mean_{}'.format(kr), dist.Delta(a11[:, idx1]))
                pyro.sample('beta_clone_n_samples_{}'.format(kr), dist.LogNormal(torch.log(prior_overdispersion), 1/prec_overdispersion))
            idx1 += 1

        if K > 0:
            with pyro.plate("subclones_{}".format(kr), K):
                adj_ccf = (subclonal_ccf * purity) / (2 * (1-purity) + theo_allele_list[kr] * purity)

                k_means = pyro.sample('beta_subclone_mean_{}'.format(kr),
                                      dist.Uniform(((subclonal_ccf - epsilon_ccf) * purity) / (2 * (1-purity) + theo_allele_list[kr] * purity),
                                                   ((subclonal_ccf + epsilon_ccf)  * purity)/ (2 * (1-purity) + theo_allele_list[kr] * purity)))

                if subclonal_prior == "Moyal":
                    scale_subclonal_param = pyro.param("scale_subclonal_{}".format(kr), torch.ones(K) * 0.02,
                                                       constraint=constraints.positive)
                    scale_subclonal = pyro.sample("scale_moyal_{}".format(kr), dist.Delta(scale_subclonal_param))
                    pyro.sample("subclones_prior_{}".format(kr),
                                                BoundedMoyal(k_means, scale_subclonal, torch.min(VAF) - 1e-5,
                                                             torch.amin(theo_peaks[kr])))
                    #pyro.sample("subclones_prior_{}".format(kr), Moyal(k_means, scale_subclonal))


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

            alpha_precision = pyro.sample('alpha_precision_{}'.format(kr), dist.Delta(alpha_precision_par[kr]))
            alpha = pyro.sample("alpha_noise_{}".format(kr),
                                dist.LogNormal(torch.log(alpha_prior * mut.theo_allele_list[karyos[kr]]),
                                               1 / alpha_precision))
            if truncated_pareto:
                if K > 0 and multi_tail:

                    pyro.sample('multitail_weights_{}'.format(kr), dist.Delta(multitail_weights[kr]).to_event(1))
                    tcm = torch.amin(theo_peaks[kr]) - torch.amax(adj_ccf)
                    tcm[tcm < (torch.min(VAF) - 1e-5)] = torch.min(VAF)
                    tcm = [tcm.detach().tolist()]
                    adccf = [(adj_ccf).detach().tolist()]
                    tcm = list(flatten([tcm, adccf]))


                    for tails in pyro.plate("subclonal_tail_{}".format(kr), K + 1):
                        if torch.Tensor(tcm)[tails] > (torch.min(VAF) + 0.05):
                            pyro.sample("tail_T_{}_{}".format(kr, tails),
                                            BoundedPareto(scale_pareto(VAF), alpha,
                                                          torch.Tensor(tcm)[tails] ))
                        else:
                            pyro.sample("tail_T_{}_{}".format(kr, tails), dist.Delta(torch.Tensor(tcm)[tails]))

                else:
                    pyro.sample("tail_T_{}".format(kr),
                                    BoundedPareto(scale_pareto(VAF), alpha,
                                                  torch.amin(theo_peaks[kr]))
                                                  )

            else:

                pyro.sample("tail_{}".format(kr), BoundedPareto(torch.min(VAF) - 1e-5, alpha, 1))

