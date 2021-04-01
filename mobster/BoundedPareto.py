from pyro.distributions import Rejector, Pareto
import torch



class BoundedPareto(Rejector):
    def __init__(self, scale, alpha, upper_limit, validate_args=None):
        propose = Pareto(scale, alpha, validate_args=validate_args)

        def log_prob_accept(x):
            return (x < upper_limit).type_as(x).log()

        #log_scale = torch.Tensor(alpha) * torch.log(torch.Tensor([scale / upper_limit]))
        log_scale = 0
        super(BoundedPareto, self).__init__(propose, log_prob_accept, log_scale)