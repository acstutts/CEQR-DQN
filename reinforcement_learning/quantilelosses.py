import torch
import numpy as np
from scipy.special import gammaln, psi

def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):

    twoBlambda = 2*beta*(1+v)
    nll = 0.5 * torch.log(torch.tensor(np.pi) / v) \
        - alpha * torch.log(twoBlambda) \
        + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)

    return torch.mean(nll) if reduce else nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5 * (a1 - 1) / b1 * (v2 * (mu2 - mu1) ** 2) \
        + 0.5 * v2 / v1 \
        - 0.5 * torch.log(torch.abs(v2) / torch.abs(v1)) \
        - 0.5 + a2 * torch.log(b1 / b2) \
        - (torch.tensor(gammaln(a1)) - torch.tensor(gammaln(a2))) \
        + (a1 - a2) * torch.tensor(psi(a1)) \
        - (b1 - b2) * a1 / b1

    return KL

def tilted_loss(q, e):
    return torch.maximum(q * e, (q - 1) * e)

def NIG_Reg(y, gamma, v, alpha, beta, quantile, omega=0.01, reduce=True, kl=False):

    error = tilted_loss(quantile, y - gamma)

    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1 + omega, beta)
        reg = error * kl
    else:
        evi = 2 * v + alpha + 1/beta
        reg = error * evi

    return torch.mean(reg) if reduce else reg

def quant_evi_loss(y_true, gamma, v, alpha, beta, quantile, coeff=1.0, reduce=True):

    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta, reduce=reduce)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta, quantile, reduce=reduce)

    return loss_nll + coeff * loss_reg

def loss_evi(y, gamma, v, alpha, beta, evi_coeff):
    
    loss_value = 0.0

    for i, q in enumerate([0.05,0.95]):
        gamma_i = gamma[:, i]
        v_i = v[:, i]
        alpha_i = alpha[:, i]
        beta_i = beta[:, i]
        
        loss_value += quant_evi_loss(y, gamma_i, v_i, alpha_i, beta_i, q, coeff=evi_coeff)
    
    return loss_value