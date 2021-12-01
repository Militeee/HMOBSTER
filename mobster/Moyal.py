import torch
import pyro
import pyro.distributions as dist
from pyro.distributions import Rejector, constraints
from pyro.distributions.torch_distribution import TorchDistribution

PI = 3.1415927410125732
EULER_MASCHERONI = 	0.57721566490153286060

class Moyal(TorchDistribution):
    has_rsample = True
    arg_constraints = {"scale": constraints.less_than(0), "loc": constraints.real}

    def __init__(self,loc, scale ,validate_args=None):
        pass

    def _standard_moyal_pdf(self, value):

        norm_const = 1/torch.sqrt(2 * PI)
        exponent = (-(value + torch.exp(-value)) / 2)

        return(torch.exp(exponent) * norm_const)

    def log_prob(self, value):
        transformed_values = (value - self._loc) / self._scale
        std_lk = self._standard_moyal_log_prob(transformed_values) / self._scale

        return(torch.log(std_lk))

    def cdf(self, value):
      return(1 - torch.erfc(torch.exp(-0.5 * value) / torch.sqrt(2)))

    @property
    def mean(self):
        return self.loc + self.scale * (EULER_MASCHERONI + torch.log(2.))

    @property
    def variance(self):
        return (self.loc**2 + PI**2) / 2

class BoundedMoyal(Rejector):
    def __init__(self, loc, scale, upper_limit, validate_args=False):
        propose = Moyal(loc, scale, validate_args=validate_args)

        def log_prob_accept(x):
            return (upper_limit >= x >= 0).type_as(x).log()

        log_scale = 0
        super(BoundedMoyal, self).__init__(propose, log_prob_accept, log_scale)