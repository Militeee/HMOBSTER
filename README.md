# HMOBSTER

Hierarchical model for neutral-evolution-aware subclonal deconvolution over different karyotypes. 
This model is a bayesian formulation and expansion of the [mobster](https://github.com/caravagnalab/mobster) algorithm ([Caravagna
et al;
PMID: 32879509](https://www.nature.com/articles/s41588-020-0675-5#:~:text=Subclonal%20reconstruction%20methods%20based%20on,and%20infer%20their%20evolutionary%20history.&text=We%20present%20a%20novel%20approach,learning%20with%20theoretical%20population%20genetics.)).

The main novelties are:
- Support for more than one karyotype at the time (mobsteh works with karyotypes 1:0, 1:1, 2:0, 2:1, 2:2)
- Full bayesian implementation with meaningfull priors (no need to bootstrap anymore)
- Hiearchical estimation of the mutation rate from different karyotypes (with credible intervals)
- Explicit separation of clonal and subclonal clusters
- GPU support and fast inference with VI and Pyro

To install just run:
`pip install HMOBSTER`

This package has very few functionalities other than the model inference itself. 
We suggest using directly the `mobsterh_fit` function in the R pakage which provides a complete interface to the python backend.

