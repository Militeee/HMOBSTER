import torch
from scipy.stats import beta, pareto, moyal

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import mobster.utils_mobster as mut


def plot_results(data, inf_res, bins=50, output = "results.png",fig_height = 4, fig_width = 3, drivers = None):

    all_params = inf_res["model_parameters"]
    tail = inf_res['run_parameters']['tail']
    K = inf_res['run_parameters']['K']
    multi_tail =  inf_res['run_parameters']['multi_tail']
    subclonal_prior = inf_res['run_parameters']['subclonal_prior']
    truncated_pareto = inf_res['run_parameters']['truncated_pareto']
    nKar = len(all_params)
    plt.rcParams["figure.figsize"] = (fig_height * nKar, fig_width * nKar)
    fig, axs = plt.subplots(nKar)
    karyos = list(data.keys())

    theoretical_num_clones = [mut.theo_clonal_num(kr, range = False) for kr in karyos]
    theo_clonal_means = [torch.min(mut.theo_clonal_num(kr)) for kr in karyos]
    for i, kr in enumerate(data):

        params = all_params[kr]
        assignment_probs = params["mixture_probs"]


        data_mut = data[kr].detach().numpy()
        data_mut = data_mut[:,0] / data_mut[:,1]

        if nKar == 1:
            axs.hist(data_mut, bins=bins, density=True, alpha=0.48)
            axs.title.set_text("Karyotype = " + kr)
        else:
            axs[i].hist(data_mut, bins=bins, density=True, alpha=0.48)
            axs[i].title.set_text("Karyotype = " + kr)




        for j in range(tail, params["beta_concentration1"].shape[0] + tail):


            cl = "tab:green"

            a = params["beta_concentration1"][j - tail]
            b = params["beta_concentration2"][j - tail]

            x = np.linspace(0.05, 1, 1000)

            tot_p = np.zeros_like(x)

            p = beta.pdf(x, a, b) * assignment_probs[j]
            tot_p += p
            if nKar == 1:

                axs.plot(x, p, linewidth=1.5, color=cl)
            else:
                axs[i].plot(x, p, linewidth=1.5, color=cl)

        if tail == 1:
            if K > 0 and truncated_pareto and multi_tail:
                for w in range(K + 1):
                    alpha = params["tail_shape"]
                    nall = mut.theo_clonal_tot(kr)
                    x = np.linspace(0.05, 1, 1000)
                    p = pareto.pdf(x, alpha * nall, scale=params["tail_scale"])
                    p[p < params["tail_higher"][w]] = 0
                    p = p / np.trapz(p,x)
                    p *= assignment_probs[0] * params["multi_tail_weights"][w]
                    tot_p += p
                    if nKar == 1:
                        axs.plot(x, p, linewidth=1.5, color="tab:pink")
                    else:
                        axs[i].plot(x, p, linewidth=1.5, color="tab:pink")

            else:
                alpha = params["tail_shape"]
                nall = mut.theo_clonal_tot(kr)
                x = np.linspace(0.05, 1, 1000)
                p = pareto.pdf(x, alpha * nall, scale=params["tail_scale"]) * assignment_probs[0]
                if truncated_pareto:
                    p[x > params["tail_higher"]] = 0
                    p = p / np.trapz(p,x)
                tot_p += p
                if nKar == 1:
                    axs.plot(x, p, linewidth=1.5, color="tab:pink")
                else:
                    axs[i].plot(x, p, linewidth=1.5, color="tab:pink")
        if drivers is not None:
            drivers_mut = data_mut[drivers[kr]]

            if(len(drivers_mut) > 0):
                if nKar == 1:
                    p = axs.patches
                    heights = [patch.get_height() for patch in p]
                    axs.vlines(drivers_mut, 0, np.max(heights))
                else:
                    p = axs[i].patches
                    heights = [patch.get_height() for patch in p]
                    axs[i].vlines(drivers_mut, 0, np.max(heights))
        for z in range(K):
            x = np.linspace(0.05, 1, 1000)

            if subclonal_prior == "Moyal":
                p = moyal.pdf(x, params["loc_subclones"][z], params["scale_subclonal"][z])
                p[x < np.min(data_mut).item()] = 0
                p[x > theo_clonal_means[i].item()] = 0
                p = p / np.trapz(p, x)
                p *= assignment_probs[(z + tail + theoretical_num_clones[i])]
                tot_p += p
            else:
                p = beta.pdf(x, params["loc_subclones"][z] * params["n_trials_subclonal"][z],
                             (1 - params["loc_subclones"][z]) * params["n_trials_subclonal"][z]) * \
                    assignment_probs[(z + tail + theoretical_num_clones[i])]
                tot_p += p

            if nKar == 1:
                axs.plot(x, p, linewidth=1.5, color="tab:red")
            else:
                axs[i].plot(x, p, linewidth=1.5, color="tab:red")

        if nKar == 1:
            axs.plot(x, tot_p, linewidth=1., linestyle = "--")
        else:
            axs[i].plot(x, tot_p, linewidth=1., linestyle = "--")
    pink_patch = mpatches.Patch(color='tab:pink', label='Tail')
    red_patch = mpatches.Patch(color='tab:red', label='Subclonal')
    green_patch = mpatches.Patch(color='tab:green', label='Clonal')

    plt.legend(handles=[pink_patch, red_patch, green_patch])

    plt.savefig(output, dpi=300)

    plt.clf()