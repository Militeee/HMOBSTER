from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions import constraints
from numbers import Number

import torch

# class BoundedPareto(Rejector):
#     def __init__(self, scale, alpha, upper_limit, validate_args=False):
#         propose = Pareto(scale, alpha, validate_args=validate_args)
#
#         def log_prob_accept(x):
#             return (x < upper_limit).type_as(x).log()
#
#         #log_scale = torch.Tensor(alpha) * torch.log(torch.Tensor([scale / upper_limit]))
#         log_scale = torch.log(Pareto(scale, alpha).cdf(upper_limit))
#         super(BoundedPareto, self).__init__(propose, log_prob_accept, log_scale)


class BoundedPareto(TorchDistribution):
    has_rsample = True
    arg_constraints = {"scale": constraints.positive, "alpha": constraints.positive,
                       "upper_limit" : constraints.positive}

    def __init__(self, scale, alpha, upper_limit, validate_args=False):

        self.scale = scale
        self.alpha = alpha

        self.upper_lim = upper_limit

        if isinstance(scale, Number) and isinstance(alpha, Number) and isinstance(upper_limit, Number):
         batch_shape = torch.Size()
        else:
         batch_shape = self.alpha.size()

        super(BoundedPareto, self).__init__(batch_shape, validate_args=validate_args)


    def ppf(self, value):

        Ha = self.upper_lim**self.alpha
        La = self.scale**self.alpha

        num = -1 * ( value * Ha - value * La - Ha )
        dem =  Ha * La

        return ( (num / dem)**(-1 / self.alpha) )


    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.alpha.dtype, device=self.alpha.device)

        return self.ppf(rand)

    def log_prob(self, value):

        mask = torch.logical_and((value < self.upper_lim),(value > self.scale)).type_as(value).log()

        num = self.alpha * self.scale**self.alpha * value**(-self.alpha - 1)
        den = 1 - (self.scale / self.upper_lim)**(self.alpha)

        return torch.log(num/den) + mask