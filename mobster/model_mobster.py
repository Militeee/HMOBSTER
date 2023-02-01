import pyro as pyro
from pyro.infer import config_enumerate
import torch
from mobster.likelihood_calculation import *
import mobster.utils_mobster as mut


@config_enumerate
def model(data, K=1, tail=1, truncated_pareto=True, subclonal_prior="Moyal", multi_tail=False, purity=0.96,
          number_of_trials_clonal_mean=500., number_of_trials_subclonal=300, number_of_trials_k=300.,
          prior_lims_clonal=[0.1, 100000.], prior_lims_k=[0.1, 100000.], alpha_precision_concentration=100,
          alpha_precision_rate=0.1, epsilon_ccf=0.01, max_min_subclonal_ccf = [0.05,0.95], k_means_init = True, min_vaf_scale_tail = 0.1):
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
        as values float torch tensors with respencìtevely NV and DP
    K : int
        Number of subclonal clusters
    tail: int
        1 if inference is to be performed with Pareto tail, 0 otherwise
    truncated_pareto: bool
        True if the pareto needs to be truncated at the mean of the lowest clonal cluster
    subclonal_prior: str
        Distribution to use for the subclonal cluster, currently supperted "Beta" and "Moyal"
    multi_tail: bool
        If True each clonal subclunal cluster has its Pareto tail, beware to run it just
        on extremely high quality data, in general prefer the False option in large-scale analysis
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


    # Here we calculate the theoretical number of clonal clusters
    theoretical_num_clones = [mut.theo_clonal_num(kr, range = False) for kr in karyos]

    # Calculate the theoretical clonal means, wihch can be analytically defined for simple karyotypes, and then multiply them by the ploidy
    theoretical_clonal_means = [mut.theo_clonal_num(kr) for kr in karyos]

    theo_allele_list = [mut.theo_clonal_tot(kr) for kr in karyos]

    # Count how many karyotypes we have in our dataset for each theoretical number of clones
    counts_clones = dict()
    for i in theoretical_num_clones:
        counts_clones[i] = counts_clones.get(i, 0) + 1

    # Prior over the mean of the alphas
    if not multi_tail:
        alpha_prior = pyro.sample('u', dist.Normal(0, 1))
    else:
        alpha_prior = pyro.sample('u', dist.Normal(0, 0.1))

    # ccf_priors = ((torch.min(torch.tensor(1) * purity) - 0.001) / (K + 1)) * torch.arange(1,K+1)
    # subclonal_ccf = pyro.sample("sb_ccf", dist.Beta(ccf_priors * number_of_trials_k, (1-ccf_priors) * number_of_trials_k))

    if K > 0:
        with pyro.plate("subclones", K):
            subclonal_ccf = pyro.sample("sb_ccf",
                                        dist.Uniform(max_min_subclonal_ccf[0], max_min_subclonal_ccf[1]))

    # We enter the karyotype plate
    # We may think about tensorizing it
    for kr in pyro.plate("kr", len(karyos)):

        NV = data[karyos[kr]][:, 0]
        DP = data[karyos[kr]][:, 1]
        VAF = NV / DP


        theo_peaks = (theoretical_clonal_means[kr] * purity - 1e-9) / (2 * (1 - purity) + theo_allele_list[kr] * purity)


        prior_overdispersion = pyro.sample('prior_overdisp_{}'.format(kr),
                                           dist.Uniform(prior_lims_clonal[0], prior_lims_clonal[1]))
        prec_overdispersion = pyro.sample('prec_overdisp_{}'.format(kr),
                                          dist.Gamma(3, 1))

        weights = pyro.sample('weights_{}'.format(kr), dist.Dirichlet(torch.ones(K + theoretical_num_clones[kr])))

        # Here we initialize both the clonal beta clusters
        with pyro.plate("clones_{}".format(kr)):


            # Number of sucessful trials for beta means prior
            bm_1 = theo_peaks.reshape([theoretical_num_clones[kr], 1]) * number_of_trials_clonal_mean

            # Number of unsucessful trials for beta means prior
            bm_2 = number_of_trials_clonal_mean - bm_1
            # As we are writing a bayesian model, beta clonal means prior are actually around
            # the theoretical values
            betas_clone_mean = pyro.sample('beta_clone_mean_{}'.format(kr), dist.Beta(bm_1, bm_2).to_event(1))



            betas_clone_n_samples = pyro.sample('beta_clone_n_samples_{}'.format(kr),
                                                dist.LogNormal(torch.log(prior_overdispersion),
                                                               1 / prec_overdispersion))

        if K > 0:
            with pyro.plate("subclones_{}".format(kr)):
                adj_ccf = (subclonal_ccf * purity) / (2 * (1 - purity) + theo_allele_list[kr] * purity)
                k_means = pyro.sample('beta_subclone_mean_{}'.format(kr),
                                      dist.Uniform(
                                          ((subclonal_ccf - epsilon_ccf) * purity) / (
                                                      2 * (1 - purity) + theo_allele_list[kr] * purity),
                                          ((subclonal_ccf + epsilon_ccf) * purity) / (
                                                      2 * (1 - purity) + theo_allele_list[kr] * purity)))


                if subclonal_prior == "Moyal":
                    scale_subclonal = pyro.sample("scale_moyal_{}".format(kr), dist.Gamma(10. * torch.ones(K), .1))
                    subclone_mean = pyro.sample("subclones_prior_{}".format(kr),
                                                BoundedMoyal(k_means - 1./scale_subclonal * (EULER_MASCHERONI + torch.log(torch.tensor(2))) , 1/scale_subclonal, torch.min(VAF) - 1e-5,
                                                             torch.min(theo_peaks)).to_event(1))

                    #subclone_mean = pyro.sample("subclones_prior_{}".format(kr),Moyal(k_means, scale_subclonal))
                else:
                    num_trials_subclonal = pyro.sample("N_subclones_{}".format(kr),
                                                       dist.Uniform(prior_lims_k[0] * torch.ones(K), prior_lims_k[1] * torch.ones(K)))
                    subclone_mean = pyro.sample("subclones_prior_{}".format(kr),
                                                dist.Beta(k_means * num_trials_subclonal,
                                                          (1 - k_means) * num_trials_subclonal))

        if (tail > 0):
            # Tail vs no tail probability, Dirichlet priors can sometimes create problems, but no better solution
            tail_probs = pyro.sample('weights_tail_{}'.format(kr), dist.Dirichlet(torch.tensor([1., 1. + K])))

            alpha_precision = pyro.sample('alpha_precision_{}'.format(kr),
                                          dist.Gamma(concentration=alpha_precision_concentration,
                                                     rate=alpha_precision_rate))
            alpha = pyro.sample("alpha_noise_{}".format(kr),
                                dist.LogNormal(alpha_prior,
                                               1 / alpha_precision))
            if truncated_pareto:
                if K > 0 and multi_tail:
                    multitails_weights = pyro.sample('multitail_weights_{}'.format(kr),
                                                     dist.Dirichlet(torch.ones(K + 1)))
                    tcm = theo_peaks - torch.amax(adj_ccf)
                    tcm[tcm < (torch.min(VAF) - 1e-5)] = torch.min(VAF)
                    adccf = [(adj_ccf).detach().tolist()]
                    tcm = list(flatten([tcm.detach().tolist(), adccf]))
                    U = tcm
                else:
                    multitails_weights = None
                    U = torch.amin(theo_peaks)
            else:
                multitails_weights = None
                U = 0.99

        with pyro.plate('data_{}'.format(kr), len(data[karyos[kr]])):

            beta = beta_lk(betas_clone_mean * betas_clone_n_samples,
                           (1 - betas_clone_mean) * betas_clone_n_samples,
                           theoretical_num_clones[kr],
                           NV, DP)
            if K > 0:
                has_subclones = True
                if subclonal_prior == "Moyal":


                    subclonal_lk = moyal_lk(subclone_mean, K, NV, DP)
                else:
                    subclonal_lk = beta_lk(subclone_mean * num_trials_subclonal,
                                           (1 - subclone_mean) * num_trials_subclonal,
                                           K, NV, DP)
            else:
                has_subclones = False

            if tail == 1:
                has_tail = True
                multi_penalty = 0
                if K > 0 and truncated_pareto and multi_tail:
                    clonal_prop = [1 - torch.amax(subclonal_ccf.detach()).item()]
                    sub_ccf = [subclonal_ccf.detach().tolist()]
                    theo_weights = torch.tensor(list(flatten([clonal_prop, sub_ccf])))
                    theo_weights = theo_weights / torch.sum(theo_weights)
                    multi_penalty = np.log(len(NV)) * torch.dist(theo_weights, multitails_weights)

                pareto = pareto_binomial_lk(alpha,U , scale_pareto(VAF, min_vaf_scale_tail), NV, DP, K, multitails_weights)
            else:
                has_tail = False
                
                

            if has_tail and has_subclones:
                not_neutral = torch.vstack([beta, subclonal_lk]) + torch.log(weights).reshape(
                    [K + theoretical_num_clones[kr], -1])
                pyro.factor("lik_{}".format(kr),
                            log_sum_exp(final_lk(pareto, not_neutral, tail_probs)).sum() + multi_penalty)

            if has_tail and not has_subclones:
                not_neutral = beta + torch.log(weights).reshape([theoretical_num_clones[kr], -1])
                pyro.factor("lik_{}".format(kr),
                            log_sum_exp(final_lk(pareto, not_neutral, tail_probs)).sum() + multi_penalty)

            if not has_tail and has_subclones:
                pyro.factor("lik_{}".format(kr),
                            log_sum_exp(torch.vstack([beta, subclonal_lk]) +
                                        torch.log(weights).reshape([K + theoretical_num_clones[kr], -1])).sum())

            if not has_tail and not has_subclones:
                pyro.factor("lik_{}".format(kr),
                            log_sum_exp(beta +
                                        torch.log(weights).reshape([theoretical_num_clones[kr], -1])).sum())
