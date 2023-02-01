from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions import constraints
from numbers import Number
import torch


class ParetoBinomial(TorchDistribution):
    has_rsample = False
    arg_constraints = {"alpha": constraints.positive,"lower_lim": constraints.positive,
                       "upper_limit": constraints.positive, "trials": constraints.positive_integer}

    def __init__(self, alpha, upper_limit, lower_lim, trials, validate_args=False):

        self.alpha = alpha
        self.lower_lim = lower_lim
        self.upper_lim = upper_limit
        self.trials = trials

        if isinstance(trials, Number) and isinstance(alpha, Number) and isinstance(upper_limit, Number) and isinstance(
                lower_lim, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.alpha.size()

        super(ParetoBinomial, self).__init__(batch_shape, validate_args=validate_args)

    def lbeta(self, a, b):
        return torch.lgamma(a + 1) + torch.lgamma(b + 1) - torch.lgamma(a + b + 1)

    def gilbeta(self, a, b, z1, z2, value):

        if z1 == 0 and z2 == 1:
            return self.lbeta(a, b)

        x = torch.linspace(z1, z2, 120).reshape([120, -1])
        y = a*torch.log(x) + b*torch.log((1 - x))
        y = y +  self.combo(self.trials, value)
        #print(y)
        inf_sup_val = torch.logsumexp(torch.logsumexp(torch.cat([y[0:-1].unsqueeze(-1),y[1:].unsqueeze(-1)],2),2),0) + torch.log((x[1]-x[0]) / 2) 
        #print(inf_sup_val)
        #print( torch.logsumexp(torch.vstack([y[0:-1], y[1:]]),0) + torch.log((x[1]-x[0]) * (len(y) - 1) / 2) )
        
        #x = torch.linspace(z1, z2, 120).reshape([120, -1])
        #y = a*torch.log(x) + b*torch.log((1 - x))
        #y = (y + self.combo(self.trials, value)).exp()
        
        


        #inf_sup_val = torch.trapz(y, x, dim=0)
        
        #return torch.log(inf_sup_val)
        return inf_sup_val 

    def combo(self, n, k):
        return (n + 1).lgamma() - (k + 1).lgamma() - ((n - k) + 1).lgamma()

    def log_prob(self, value):

        K =  torch.log(self.alpha) + torch.log(self.lower_lim ** self.alpha) + torch.log(
                    torch.tensor(1)) - torch.log(1 - (self.lower_lim / self.upper_lim) ** self.alpha)

        integral = self.gilbeta(value - self.alpha - 1, self.trials - value, self.lower_lim, self.upper_lim, value)

        res = K + integral

        return res
