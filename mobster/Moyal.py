import torch
import pyro
import pyro.distributions as dist
from pyro.distributions import Rejector, constraints
from pyro.distributions.torch_distribution import TorchDistribution
from numbers import Number


PI = 3.1415927410125732
EULER_MASCHERONI = 	0.57721566490153286060

class Moyal(TorchDistribution):
    has_rsample = True
    arg_constraints = {"scale": constraints.positive, "loc": constraints.real}

    def __init__(self,loc, scale ,validate_args=False):

        self.loc = torch.tensor(loc).float()
        self.scale = torch.tensor(loc).float()

        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Moyal, self).__init__(batch_shape, validate_args=validate_args)

    def _standard_moyal_pdf(self, value):

        norm_const = 1/torch.sqrt(2 * PI)
        exponent = (-(value + torch.exp(-value)) / 2)

        return(torch.exp(exponent) * norm_const)

    def log_prob(self, value):
        transformed_values = (value - self._loc) / self._scale
        std_lk = self._standard_moyal_log_prob(transformed_values) / self._scale

        return(torch.log(std_lk))

    def cdf(self, value):
      return(1 - torch.erfc(torch.exp(-0.5 * (value - self.loc) / self.scale ) / torch.sqrt(2)))

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Moyal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Moyal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def ppf(self, value):
        inv_err = -dist.Normal(0,1).icdf(value * 0.5) * torch.sqrt(torch.tensor(1/2))
        return -torch.log(2 * inv_err ** 2) * self.scale + self.loc

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.ppf(rand)

    @property
    def mean(self):
        return self.loc + self.scale * (EULER_MASCHERONI + torch.log(2.))

    @property
    def variance(self):
        return (self.loc**2 + PI**2) / 2

class BoundedMoyal(Rejector):
    def __init__(self, loc,scale, lower_limit, upper_limit, validate_args=False):
        propose = Moyal(loc, scale, validate_args=validate_args)

        def log_prob_accept(x):
            return (upper_limit >= x >= 0).type_as(x).log()

        log_scale = torch.log(propose.cdf(upper_limit) - propose.cdf(lower_limit))
        super(BoundedMoyal, self).__init__(propose, log_prob_accept, log_scale)