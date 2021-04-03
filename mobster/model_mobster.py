import pyro as pyro
from pyro.infer import config_enumerate
import torch
from mobster.likelihood_calculation import *
import mobster.utils_mobster as mut



@config_enumerate
def model(data, K=1, tail=1, truncated_pareto = True, purity=0.96,  number_of_trials_clonal_mean=500., number_of_trials_k=300.,
         prior_lims_clonal=[0.1, 100000.], prior_lims_k=[0.1, 100000.], alpha_precision_concentration = 5, alpha_precision_rate=0.1):

    """Hierarchical bayesian model for Subclonal Deconvolution from VAF

    This model deconvolves the signal from the Variant Allelic Frequency spectrum.


    Parameters
    ----------
    data : dictionary
    A dictionary with karyotypes as keys (written in this form major_allele:minor_allele) and
    as values float torch tensors with the VAF value
    K : int
    Number of subclonal clusters
    tail: int
    1 if inference is to perform with Pareto tail, 0 otherwise
    purity: float
    Previously estimated purity of the tumor
    alpha_prior_sd: float
    Prior standard deviation for the LogNormal distribution describing
    the shape of the Pareto Distribution
    number_of_trials_clonal_mean : int
    Number of trials for the clonal
    prior_lims_clonal : list
    Limits of the Uniform over the prior for the number of trials for the clonal clusters
    prior_lims_k : list
    Limits of the Uniform over the prior for the number of trials for the subclonal clusters

    """

    # Split major and minor allele
    karyos = list(data.keys())

    # Initialize means as they are into conditional block
    betas_subclone_mean2 = 0
    betas_subclone_n_samples2 = 0
    betas_subclone_mean = 0
    betas_subclone_n_samples = 0

    # Here we calculate the theoretical number of clonal clusters
    theoretical_num_clones = [mut.theo_clonal_list[kr] for kr in karyos]

    # Calculate the theoretical clonal means, wihch can be analytically defined for simple karyotypes, and then multiply them by the ploidy
    theoretical_clonal_means = [mut.theo_clonal_means_list[kr] * purity for kr in karyos]


    # Count how many karyotypes we have in our dataset for each theoretical number of clones
    counts_clones = dict()
    for i in theoretical_num_clones:
        counts_clones[i] = counts_clones.get(i, 0) + 1

    # Prior over the mean of the alphas
    alpha_prior = pyro.sample('u', dist.Uniform(0.1, 5))


    # We enter the karyotype plate
    # We may think about tensorizing it
    for kr in pyro.plate("kr", len(karyos)):



        # We construct two different computational graphs for cases with 2 and 1 clonal clusters
        if theoretical_num_clones[kr] == 2:

            # mixture weights for the beta parameters with theoretical number of clones 2
            weights_2 = pyro.sample('weights_{}'.format(kr), dist.Dirichlet(torch.ones(K + 2)))

            # Here we initialize both the clonal and the subclonal beta clusters
            with pyro.plate("clones_{}".format(kr), 2 + K):

                # Subclonal clusters are initialized halfway between zero and the smallest clonal cluster
                if K == 0:
                    k_means = torch.zeros(0)
                else:
                    k_means = ((torch.min(theoretical_clonal_means[kr] * purity) - 0.01) / (K + 1)) * torch.arange(1,K+1)
                # Number of sucessful trials for beta means prior
                bm_12 = torch.tensor(
                    flatten_list([theoretical_clonal_means[kr].tolist(), k_means.tolist()])) * number_of_trials_clonal_mean
                # Number of unsucessful trials for beta means prior
                bm_22 = number_of_trials_clonal_mean - bm_12
                # As we are writing a bayesian model, beta clonal means prior are actually around
                # the theoretical values
                betas_subclone_mean2 = pyro.sample('beta_clone_mean_{}'.format(kr), dist.Beta(bm_12, bm_22))

                # Here we prepare the prior for the beta number of samples
                bns_12 = torch.tensor((2 * [prior_lims_clonal[0]]) + (K * [prior_lims_k[0]]))
                bns_22 = torch.tensor((2 * [prior_lims_clonal[1]]) + (K * [prior_lims_k[1]]))
                betas_subclone_n_samples2 = pyro.sample('beta_clone_n_samples_{}'.format(kr),
                                                        dist.Uniform(bns_12, bns_22))
        # The same as before but with number of clonal cluster equal to 1
        else:

            weights_1 = pyro.sample('weights_{}'.format(kr), dist.Dirichlet(torch.ones(K + 1)))
            # Pytorch kinda sucks sometimes, so some code here is pretty hard to read
            # Maybe to rewrite using only numpy arrays and not lists
            with pyro.plate("clones_{}".format(kr), 1 + K):

                if K == 0:
                    k_means = torch.zeros(0)
                else:
                    k_means = (theoretical_clonal_means[kr] - 0.01) / (K + 1) * torch.arange(1,K+1)
                bm_11 = torch.tensor(
                    flatten_list([theoretical_clonal_means[kr].tolist(), k_means.tolist()])) * number_of_trials_clonal_mean
                bm_21 = number_of_trials_clonal_mean - bm_11
                betas_subclone_mean = pyro.sample('beta_clone_mean_{}'.format(kr), dist.Beta(bm_11, bm_21))
                bns_11 = torch.tensor((1 * [prior_lims_clonal[0]]) + (K * [prior_lims_k[0]]))
                bns_21 = torch.tensor((1 * [prior_lims_clonal[1]]) + (K * [prior_lims_k[1]]))
                betas_subclone_n_samples = pyro.sample('beta_clone_n_samples_{}'.format(kr),
                                                       dist.Uniform(bns_11, bns_21))
        if (tail == 1):
            # Tail vs no tail probability, Dirichlet priors can sometimes create problems, but no better solution
            tail_probs = pyro.sample('weights_tail_{}'.format(kr), dist.Dirichlet(torch.ones(2)))
        with pyro.plate('data_{}'.format(kr), len(data[karyos[kr]])):

        # Here we split again the computational graph in case of tail or no tail
            if (tail == 1):

                alpha_precision = pyro.sample('alpha_precision_{}'.format(kr),
                                              dist.Gamma(concentration=alpha_precision_concentration,
                                                         rate=alpha_precision_rate))

                alpha = pyro.sample("alpha_noise_{}".format(kr),
                                    dist.LogNormal(torch.log(alpha_prior * mut.theo_allele_list[karyos[kr]]), 1 / alpha_precision))

                # The likelihood is different among the karyotypes classes
                if theoretical_num_clones[kr] == 2:
                    beta = beta_lk(betas_subclone_mean2 * betas_subclone_n_samples2,
                                   (1 - betas_subclone_mean2) * betas_subclone_n_samples2,
                                   weights_2, K + theoretical_num_clones[kr],
                                   data[karyos[kr]])
                    if truncated_pareto:
                        pareto = BoundedPareto(torch.min(data[karyos[kr]]) - 1e-5, alpha, torch.amin(theoretical_clonal_means[kr])).log_prob(data[karyos[kr]])
                    else:
                        pareto = dist.Pareto(torch.min(data[karyos[kr]]) - 1e-5, alpha).log_prob(data[karyos[kr]])

                    pyro.factor("lik_{}".format(kr), log_sum_exp(final_lk(pareto, beta, tail_probs)).sum())


                else:
                    beta = beta_lk(betas_subclone_mean * betas_subclone_n_samples,
                                   (1 - betas_subclone_mean) * betas_subclone_n_samples,
                                   weights_1, K + theoretical_num_clones[kr],
                                   data[karyos[kr]])
                    if truncated_pareto:
                        pareto = BoundedPareto(torch.min(data[karyos[kr]]) - 1e-5, alpha, torch.amin(theoretical_clonal_means[kr])).log_prob(data[karyos[kr]])
                    else:
                        pareto = dist.Pareto(torch.min(data[karyos[kr]]) - 1e-5, alpha).log_prob(data[karyos[kr]])
                    pyro.factor("lik_{}".format(kr), log_sum_exp(final_lk(pareto, beta, tail_probs)).sum())

            else:

                if theoretical_num_clones[kr] == 2:
                    pyro.factor("lik_{}".format(kr), torch.sum(log_sum_exp(beta_lk(betas_subclone_mean2 * betas_subclone_n_samples2,
                                                                       (1 - betas_subclone_mean2) * betas_subclone_n_samples2,
                                                                       weights_2, K + theoretical_num_clones[kr],
                                                                       data[karyos[kr]]))))
                else:
                    pyro.factor("lik_{}".format(kr), torch.sum(log_sum_exp(beta_lk(betas_subclone_mean * betas_subclone_n_samples,
                                                                       (1 - betas_subclone_mean) * betas_subclone_n_samples,
                                                                       weights_1,
                                                                       K + theoretical_num_clones[kr],
                                                                       data[karyos[kr]]))))



