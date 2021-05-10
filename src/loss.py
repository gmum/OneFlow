import numpy as np
import torch
import torch.nn as nn

from scipy.special import binom


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## OneFlow loss

def binomial_coefficient(n, k, p):
    return binom(n, k) * (p ** k) * ((1-p) ** (n-k))


def sample_from_n_ball(n, size):
    s = np.random.normal(size=(size, n)) 
    s /= np.linalg.norm(s, axis=1)[:, np.newaxis]

    u = np.random.uniform(low=0.0, high=1.0, size=size)
    r = np.power(u, 1./n)

    r = torch.from_numpy(r[:, np.newaxis] * s)
    return r


def binomial_loss_const_det(z, p):
    n = z.shape[0]

    ### We take r^2
    r = torch.sqrt(torch.sum(z ** 2, dim=1))
    sorted_r = torch.sort(r, descending=True)[0]

    coeff = binomial_coefficient(n, np.arange(1, n+1), p)
    coeff = torch.from_numpy(coeff).to(device)

    return torch.dot(sorted_r, coeff)


def binomial_loss_var_det(z, logdetJ, p, model):
    n = z.shape[0]
    data_dim = z.shape[1]
    
    r = binomial_loss(z, p)
    ball_samples = sample_from_n_ball(data_dim, n).to(device)
    
    model.eval()
    with torch.no_grad():
        y = model.inv_flow(r * ball_samples)
        _, logdetJ = model.flow(y)
        
    return data_dim * r + torch.logsumexp(1. / logdetJ.flatten(), 0)


class OneFlowLoss(nn.Module):
    """
    Implements the loss of a OneFlow model.
    """
    def __init__(self, p, model):
        super(OneFlowLoss, self).__init__()
        
        self.p = p
        self.model = model
        self.det_type = self.model.det_type
        
        # Numerical variables
        self.total_loss = None
        

    def forward(self, z, logdetJ):
        # Compute pytorch loss
        if self.det_type == 'const':
            tot_loss = binomial_loss_const_det(z, self.p)
        elif self.det_type == 'var':
            tot_loss = binomial_loss_var_det(z, logdetJ, self.p, self.model)

        # Store numerical
        self.total_loss = tot_loss

        return tot_loss


## Flow loss

def ll_loss(z, logdetJ, prior_z):
    z = z.view(z.shape[0], -1).float()
    ll_z = prior_z.log_prob(z.cpu()).double().to(device).view(-1,1) + logdetJ
    return -torch.mean(ll_z).double()


class FlowLoss(nn.Module):
    """
    Implements the loss of a LL-Flow model.
    """
    def __init__(self, prior_z):
        super(FlowLoss, self).__init__()
        
        self.prior_z = prior_z
        
        # Numerical variables
        self.total_loss = None
        

    def forward(self, z, logdetJ):
        # Compute pytorch loss
        tot_loss = ll_loss(z, logdetJ, self.prior_z)

        # Store numerical
        self.total_loss = tot_loss

        return tot_loss
