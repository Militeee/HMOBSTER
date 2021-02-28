from scipy.stats import beta, pareto

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_results(data, inf_res, bins=50, output = "results.png",fig_height = 10, fig_width = 8):

    all_params = inf_res["model_parameters"]
    plt.rcParams["figure.figsize"] = (fig_height, fig_width)
    tail = inf_res['run_parameters']['tail']
    fig, axs = plt.subplots(len(all_params))
    for i, kr in enumerate(data):

        axs[i].hist(data[kr].detach().numpy(), bins=bins, density=True, alpha=0.48)
        axs[i].title.set_text("Karyotype = " + kr)
        params = all_params[kr]

        karyos = list(data.keys())

        major = [int(str(i).split(":")[0]) for i in karyos]
        minor = [int(str(i).split(":")[1]) for i in karyos]
        theoretical_num_clones = [1 if (mn == 0 or mn == mj) else 2 for mj, mn in zip(major, minor)]
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


            axs[i].plot(x, p, linewidth=3, color=cl)

        if tail == 1:
            alpha = params["tail_shape"]
            x = np.linspace(0.05, 1, 200)
            p = pareto.pdf(x, alpha, scale=params["tail_scale"]) * assignment_probs[0]
            axs[i].plot(x, p, linewidth=3, color="tab:pink")


    pink_patch = mpatches.Patch(color='tab:pink', label='Tail')
    red_patch = mpatches.Patch(color='tab:red', label='Subclonal')
    green_patch = mpatches.Patch(color='tab:green', label='Clonal')

    plt.legend(handles=[pink_patch, red_patch, green_patch])

    plt.savefig(output, dpi=300)