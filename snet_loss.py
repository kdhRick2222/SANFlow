from curses import A_DIM
from zlib import Z_NO_COMPRESSION
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi, log
from random import randint
from gamma import LogGamma

cross_entropy = nn.CrossEntropyLoss()
log_gamma = LogGamma.apply

np.random.seed(0)
L1_loss = nn.L1Loss()
_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))
log_2 = 0.6931471805599453
eps2 = 1e-2
# clip bound
log_max = log(1e4)
log_min = log(1e-8)
softmax = nn.Softmax()

class Score_Observer:
    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = 0.0
        self.last = 0.0

    def update(self, score, epoch, print_score=True):
        self.last = score
        save_weights = False
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
            save_weights = True
        if print_score:
            self.print_score()
        
        return save_weights

    def print_score(self):
        print('{:s}: \t last: {:.1f} \t max: {:.1f} \t epoch_max: {:d}'.format(
            self.name, self.last, self.max_score, self.max_epoch))


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


log_theta = torch.nn.LogSigmoid()


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def get_logp_snet_mean(epoch, C, z, logdet_J, mask, log_alpha, log_beta):
    
    var = torch.exp(log_beta) / (torch.exp(log_alpha.clamp_(min=log_min, max=log_max)) + 1)
    var = var.clamp_(min=0, max=5.0)
    var_detach = var.detach()
    
    if len(mask) == 1:
        alpha = torch.exp(log_alpha.clamp_(min=log_min, max=log_max))
        alpha_div_beta = torch.exp(log_alpha - log_beta)

        # logp_n = C*_GCONST_ - C*(log_beta.squeeze(1) - torch.digamma(alpha.squeeze(1))) - 0.5 * torch.sum(z**2, 1) * alpha_div_beta.squeeze(1)
        logp_n =  C*_GCONST_ - 0.5*torch.sum(z**2, 1)/var.squeeze(1) - C*0.5*torch.log(var.squeeze(1)) #original

        logp = logp_n + logdet_J

        return logp

    else:
        mask_a = mask.clip(0, 1)
        mask_n = 1 - mask_a
        ones = mask_n + mask_a
        mask_a_r = torch.ceil(mask)
        mask_n_r = 1 - mask_a_r
        p = 11.
        alpha_ = torch.exp(log_alpha.clamp_(min=log_min, max=log_max))
        alpha = alpha_

        alpha0 = ones * p**2 * 0.5 - 1.
        beta0 = ones * p**2 * 0.1 * 0.5

        log_alpha_ = log_alpha * mask_n_r + log_alpha.detach() * mask_a_r
        log_beta_ = log_beta * mask_n_r + log_beta.detach() * mask_a_r

        alpha_div_beta = torch.exp(log_alpha_ - log_beta_)
        alpha_div_beta_ = torch.exp(log_alpha - log_beta)

        logp_n2 = C*_GCONST_ - C*(log_beta_.squeeze(1) - torch.digamma(alpha.squeeze(1))) - 0.5 * torch.sum(z**2, 1) * alpha_div_beta.squeeze(1)
        # logp_a2 = C*_GCONST_ - C*(log_beta_.detach().squeeze(1) - torch.digamma(alpha.detach().squeeze(1))) - 0.5 * torch.sum((z-1)**2, 1) * alpha_div_beta.detach().squeeze(1)
        logp_a2 = C*_GCONST_ - 0.5*torch.sum((z-1)**2, 1)/var_detach.squeeze(1) - C*0.5*torch.log(var_detach.squeeze(1)) #original

        logp_2 = logp_n2 * mask_n_r.squeeze(1) + logp_a2 * mask_a_r.squeeze(1)
        
        # # cross-entropy
        log_sum = torch.cat((logp_n2.unsqueeze(1), logp_a2.unsqueeze(1)), dim = 1)
        loss_ce = F.cross_entropy(torch.softmax(log_sum, 1), mask_a_r.squeeze(1).long())

        logp = logp_2 + logdet_J

        # # Inverse Gamma
        kl_Igamma = torch.mean((alpha - alpha0)*torch.digamma(alpha) + (log_gamma(alpha0) - log_gamma(alpha))
                               + alpha0*(log_beta_ - torch.log(beta0)) + beta0 * alpha_div_beta - alpha, 1)

        loss = logp - kl_Igamma - 2e-1*loss_ce

        return loss

