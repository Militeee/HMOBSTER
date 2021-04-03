import pyro as pyro
import numpy as np
from pyro.infer import MCMC, NUTS, Predictive, HMC, config_enumerate
from pyro.infer.autoguide import AutoDelta, init_to_sample
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from torch.distributions import constraints
import mobster.utils_mobster as mut


@config_enumerate
def guide(data, K=1, tail=1, truncated_pareto = True, purity=0.96, clonal_beta_var=1., number_of_trials_clonal_mean=100.,
          number_of_trials_clonal=900., number_of_trials_k=300., prior_lims_clonal=[1., 10000.],alpha_precision_concentration = 5, alpha_precision_rate=0.1,
          prior_lims_k=[1., 10000.]):
    """
    Parameters
    ----------
    param1 :
    The first parameter.
    param2 :
    The second parameter.

    """
    karyos = list(data.keys())

    # Here we calculate the theoretical number of clonal clusters
    theoretical_num_clones = [mut.theo_clonal_list[kr] for kr in karyos]

    # Calculate the theoretical clonal means, wihch can be analytically defined for simple karyotypes, and then multiply them by the ploidy
    theoretical_clonal_means = [mut.theo_clonal_means_list[kr] * purity for kr in karyos]

    counts_clones = dict()
    for i in theoretical_num_clones:
        counts_clones[i] = counts_clones.get(i, 0) + 1

    index_2 = [i for i, j in enumerate(theoretical_num_clones) if j == 2]
    index_1 = [i for i, j in enumerate(theoretical_num_clones) if j == 1]

    a_prior = pyro.param("tail_mean", torch.tensor(1), constraint=constraints.interval(0.5, 5))
    alpha_precision_par = pyro.param("alpha_noise",
                                 dist.Gamma(concentration=alpha_precision_concentration, rate=alpha_precision_rate).mean * torch.ones([len(karyos)]),
                                 constraint=constraints.positive)





    alpha_prior = pyro.sample('u', dist.Delta(a_prior))

    idx1 = 0
    idx2 = 0

    if tail == 1:
        weights_tail = pyro.param("param_tail_weights", 1 / torch.ones([len(karyos), 2]), constraint=constraints.simplex)

    if 2 in theoretical_num_clones:
        weights_param_2 = pyro.param("param_weights_2", (1 / (K + 2)) * torch.ones([len(index_2), K + 2]),
                                     constraint=constraints.simplex)
    if 1 in theoretical_num_clones:
        weights_param_1 = pyro.param("param_weights_1", (1 / (K + 1)) * torch.ones([len(index_1), K + 1]),
                                     constraint=constraints.simplex)

    if K != 0:
        if 2 in counts_clones.keys():

            k2 = torch.cat([theoretical_clonal_means[kr] * purity for kr in range(len(karyos)) if theoretical_num_clones[kr] == 2]).reshape([counts_clones[2],2])


            k_means_2 = ((torch.amin(k2, dim = 1) - 0.01) / (K +1) ).reshape([1, counts_clones[2]]) * torch.arange(1, K + 1).reshape([K,1]) + 0.02
        else:
            k_means_2 = 0



        if 1 in counts_clones.keys():
            k1 = torch.tensor([theoretical_clonal_means[kr] * purity for kr in range(len(karyos)) if theoretical_num_clones[kr] == 1]).reshape([counts_clones[1],1])

            k_means_1 = ((torch.amin(k1, dim = 1) - 0.01) / (K + 1)).reshape([1,counts_clones[1]]) * torch.arange(1, K + 1).reshape([K, 1]) + 0.02
        else:
            k_means_1 = 0
    else:
        k_means_1 = k_means_2 = 0



    for kr in pyro.plate("kr", len(karyos)):



        if theoretical_num_clones[kr] == 2:

            pyro.sample('weights_{}'.format(kr), dist.Delta(weights_param_2[idx2]).to_event(1))

            # Mean parameter when the number of clonal picks is 2
            a_2_theo = torch.cat([theoretical_clonal_means[i] for i in index_2]).reshape([2, counts_clones[2]])

            # Number of trials parameter when the number of clonal picks is 2
            b_2_theo = torch.ones([2, len(index_2)]) * number_of_trials_clonal_mean

            # If we have subclones
            a21 = a22 = 0
            if K != 0:
                # mean of the subclones, located more or less between the smallest clonal cluster and the tail
                a_2_k = dist.Uniform(k_means_2 - 0.05, k_means_2 + 0.05).sample().reshape([K, counts_clones[2]])
                # Number of trials  for the subclones
                b_2_k = torch.ones([K, len(index_2)]) * number_of_trials_k

                a21 = pyro.param('a_2',
                                 torch.cat((a_2_theo, a_2_k)).reshape([2 + K, len(index_2)]),
                                 constraint=constraints.unit_interval)
                a22 = pyro.param('b_2',
                                 torch.cat((b_2_theo, b_2_k)).reshape([2 + K, len(index_2)]),
                                 constraint=constraints.positive)
            else:
                a21 = pyro.param('a_2',
                                 a_2_theo.reshape([2 + K, len(index_2)]),
                                 constraint=constraints.unit_interval)
                a22 = pyro.param('b_2',
                                 b_2_theo.reshape([2 + K, len(index_2)]),
                                 constraint=constraints.positive)



            with pyro.plate("clones_{}".format(kr), 2 + K):
                pyro.sample('beta_clone_mean_{}'.format(kr), dist.Delta(a21[:, idx2]))
                pyro.sample('beta_clone_n_samples_{}'.format(kr), dist.Delta(a22[:, idx2]))
            idx2 += 1

        else:
            pyro.sample('weights_{}'.format(kr), dist.Delta(weights_param_1[idx1]).to_event(1))

            # Mean parameter when the number of clonal picks is 1
            a_1_theo = torch.tensor([theoretical_clonal_means[i] for i in index_1]).reshape([1, counts_clones[1]])

            # Number of trials parameter when the number of clonal picks is 1
            b_1_theo = torch.ones([1, len(index_1)]) * number_of_trials_clonal_mean

            if K != 0:

                a_1_k = dist.Uniform(k_means_1 - 0.05, k_means_1 + 0.05).sample().reshape([K, counts_clones[1]])
                b_1_k = torch.ones([K, len(index_1)]) * number_of_trials_k

                a11 = pyro.param('a_1',
                                 torch.cat((a_1_theo, a_1_k)).reshape([1 + K, len(index_1)]),
                                 constraint=constraints.unit_interval)
                a12 = pyro.param('b_1',
                                 torch.cat((b_1_theo, b_1_k)).reshape([1 + K, len(index_1)]),
                                 constraint=constraints.positive)
            else:


                a11 = pyro.param('a_1',
                                 a_1_theo.reshape([1 + K, len(index_1)]),
                                 constraint=constraints.unit_interval)
                a12 = pyro.param('b_1',
                                 b_1_theo.reshape([1 + K, len(index_1)]),
                                 constraint=constraints.positive)

            with pyro.plate("clones_{}".format(kr), 1 + K):
                pyro.sample('beta_clone_mean_{}'.format(kr), dist.Delta(a11[:, idx1]))
                pyro.sample('beta_clone_n_samples_{}'.format(kr), dist.Delta(a12[:, idx1]))
            idx1 += 1

        if tail == 1:
            # K = K + tail
            pyro.sample('weights_tail_{}'.format(kr), dist.Delta(weights_tail[kr]).to_event(1))
            alpha_precision = pyro.sample('alpha_precision_{}'.format(kr), dist.Delta(alpha_precision_par[kr]))

            with pyro.plate('data_{}'.format(kr), len(data[karyos[kr]])):
                pyro.sample("alpha_noise_{}".format(kr),
                            dist.LogNormal(torch.log(alpha_prior) * mut.theo_allele_list[karyos[kr]], 1 / alpha_precision))

