import pyro as pyro
from pyro.infer import config_enumerate
import torch
from mobster.likelihood_calculation import *
import mobster.utils_mobster as mut




@config_enumerate
def model(data, K=1, tail=1, truncated_pareto = True, purity=0.96,  number_of_trials_clonal_mean=500., number_of_trials_k=300.,
         prior_lims_clonal=[0.1, 100000.], prior_lims_k=[0.1, 100000.], alpha_precision_concentration = 5, alpha_precision_rate=0.1, epsilon_ccf = 0.01):

    """Hierarchical bayesian model for Subclonal Deconvolution

    This model deconvolves the signal from the Variant Allelic Frequency (VAF) spectrum using a sound
    evolutionary model based approach. The model is basically a hierarchical mixture of Beta distributions with the
    possibility of adding a Pareto (can be also truncated) to the final mixture. Biologically the Betas should model
    clonal and subclonal picks while the Pareto accounts for neutral Tails. It works on different karyotypes
    (at the moment just few are supported), anyway this function should be used preferably with the `R interface <https://github.com/Militeee/rcongas>`_
    as it provides all the necessary checks.



    Parameters
    ----------
    data : dictionary
        A dictionary with karyotypes as keys (written in this form major_allele:minor_allele) and
        as values float torch tensors with the VAF value
    K : int
        Number of subclonal clusters
    tail: int
        1 if inference is to perform with Pareto tail, 0 otherwise
    truncated_pareto: bool
        True if the pareto needs to be truncated at the mean of the lowest clonal cluster
    purity: float
        Previously estimated purity of the tumor
    number_of_trials_clonal_mean: float
        Number of trials for the clonal betas prior
    number_of_trials_k : float
        Number of trials for the subclonal betas prior
    prior_lims_clonal : list
        limits for the Uniform prior for the number of the clonal Betas number of trials
    prior_lims_k : list
        Limits of the Uniform over the prior of the number of trials for the subclonal clusters
    alpha_precision_concentration: float
        Hyperprior for the concentration parameter of the Gamma distribution modelling alpha precision
    alpha_precision_rate: float
        Hyperprior for the rate parameter of the Gamma distribution modelling alpha precision
    epsilon_ccf : float
        Tolerance over CCF peak in a single karyotype

        


    Notes
    -----

    Note that Beta distributions here are parametrized as :math:`Beta(\alpha * T, (1-\alpha) * T)`, where :math:`T` is the
    numer of trials and :math:`\alpha` the success probability


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

    #ccf_priors = ((torch.min(torch.tensor(1) * purity) - 0.001) / (K + 1)) * torch.arange(1,K+1)
    #subclonal_ccf = pyro.sample("sb_ccf", dist.Beta(ccf_priors * number_of_trials_k, (1-ccf_priors) * number_of_trials_k))

    subclonal_ccf = pyro.sample("sb_ccf",
                                dist.Uniform(0.000001, 0.99999))

    # We enter the karyotype plate
    # We may think about tensorizing it
    for kr in pyro.plate("kr", len(karyos)):

        adj_ccf = subclonal_ccf * mut.ccf_adjust[karyos[kr]]
        k_means = pyro.sample('beta_subclone_mean_{}'.format(kr),
                              dist.Uniform(adj_ccf - epsilon_ccf, adj_ccf + epsilon_ccf))

        # We construct two different computational graphs for cases with 2 and 1 clonal clusters
        if theoretical_num_clones[kr] == 2:

            # mixture weights for the beta parameters with theoretical number of clones 2
            weights_2 = pyro.sample('weights_{}'.format(kr), dist.Dirichlet(torch.ones(K + 2)))

            # Here we initialize both the clonal beta clusters
            with pyro.plate("clones_{}".format(kr), 2):

                # Number of sucessful trials for beta means prior
                bm_12 = torch.tensor(theoretical_clonal_means[kr].tolist()) * number_of_trials_clonal_mean

                # Number of unsucessful trials for beta means prior
                bm_22 = number_of_trials_clonal_mean - bm_12
                # As we are writing a bayesian model, beta clonal means prior are actually around
                # the theoretical values
                betas_subclone_mean2 = pyro.sample('beta_clone_mean_{}'.format(kr), dist.Beta(bm_12, bm_22))


                if K != 0:
                    betas_subclone_mean2 = torch.cat([betas_subclone_mean2,k_means])

            # Here we initialize both the clonal and the subclonal beta clusters
            with pyro.plate("clones_N_{}".format(kr), 2 + K):
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
            with pyro.plate("clones_{}".format(kr), 1):



                bm_11 = theoretical_clonal_means[kr] * number_of_trials_clonal_mean
                bm_21 = number_of_trials_clonal_mean - bm_11
                betas_subclone_mean = pyro.sample('beta_clone_mean_{}'.format(kr), dist.Beta(bm_11, bm_21))

                if K != 0:
                    betas_subclone_mean = torch.cat([betas_subclone_mean,k_means])

            with pyro.plate("clones_N_{}".format(kr), 1 + K):
                bns_11 = torch.tensor((1 * [prior_lims_clonal[0]]) + (K * [prior_lims_k[0]]))
                bns_21 = torch.tensor((1 * [prior_lims_clonal[1]]) + (K * [prior_lims_k[1]]))
                betas_subclone_n_samples = pyro.sample('beta_clone_n_samples_{}'.format(kr),
                                                       dist.Uniform(bns_11, bns_21))

        if (tail == 1):
            # Tail vs no tail probability, Dirichlet priors can sometimes create problems, but no better solution
            tail_probs = pyro.sample('weights_tail_{}'.format(kr), dist.Dirichlet(torch.ones(2)))
            alpha_precision = pyro.sample('alpha_precision_{}'.format(kr),
                                          dist.Gamma(concentration=alpha_precision_concentration,
                                                     rate=alpha_precision_rate))
        with pyro.plate('data_{}'.format(kr), len(data[karyos[kr]])):

        # Here we split again the computational graph in case of tail or no tail
            if (tail == 1):


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



