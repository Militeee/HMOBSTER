from scipy.stats import beta, pareto, lognorm

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import mobster.utils_mobster as mut


def plot_results(data, inf_res, bins=50, output = "results.png",fig_height = 4, fig_width = 3, drivers = None):

    all_params = inf_res["model_parameters"]
    tail = inf_res['run_parameters']['tail']
    nKar = len(all_params)
    plt.rcParams["figure.figsize"] = (fig_height * nKar, fig_width * nKar)
    fig, axs = plt.subplots(nKar)
    for i, kr in enumerate(data):
        data_mut = data[kr].detach().numpy()
        if nKar == 1:
            axs.hist(data_mut, bins=bins, density=True, alpha=0.48)
            axs.title.set_text("Karyotype = " + kr)
        else:
            axs[i].hist(data_mut, bins=bins, density=True, alpha=0.48)
            axs[i].title.set_text("Karyotype = " + kr)
        params = all_params[kr]

        karyos = list(data.keys())

        major = [int(str(i).split(":")[0]) for i in karyos]
        minor = [int(str(i).split(":")[1]) for i in karyos]
        theoretical_num_clones = [mut.theo_clonal_list[kr] for kr in karyos]
        assignment_probs = params["mixture_probs"]

        for j in range(tail, len(assignment_probs)):

            if theoretical_num_clones[i] == 1:
                cl = "tab:green" if (j - tail) < 1 else "tab:red"

            else:
                cl = "tab:green" if (j - tail) < 2 else "tab:red"

            a = params["beta_concentration1"][j - tail]
            b = params["beta_concentration2"][j - tail]

            x = np.linspace(0.05, 1, 200)

            p = beta.pdf(x, a, b) * assignment_probs[j]
            if nKar == 1:
                axs.plot(x, p, linewidth=1.5, color=cl)
            else:
                axs[i].plot(x, p, linewidth=1.5, color=cl)

        if tail == 1:
            alpha = params["tail_shape"]
            nall = mut.theo_allele_list[kr]
            x = np.linspace(0.05, 1, 200)
            p = pareto.pdf(x, alpha * nall, scale=params["tail_scale"]) * assignment_probs[0]
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


    pink_patch = mpatches.Patch(color='tab:pink', label='Tail')
    red_patch = mpatches.Patch(color='tab:red', label='Subclonal')
    green_patch = mpatches.Patch(color='tab:green', label='Clonal')

    plt.legend(handles=[pink_patch, red_patch, green_patch])

    plt.savefig(output, dpi=300)

    plt.clf()