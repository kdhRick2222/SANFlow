import numpy as np
import torch
import torch.nn.functional as F
from random import randint
import math

np.random.seed(0)
_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))
log_2 = 0.6931471805599453


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
        print('{:s}: \t last: {:.2f} \t max: {:.2f} \t epoch_max: {:d}'.format(
            self.name, self.last, self.max_score, self.max_epoch))


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


# def get_logp(C, z, logdet_J, mask):
#     # print(mask.sum())
#     # print(len(mask))
#     # z.size() == torch.Size([256,1])
#     # mask.size() == torch.Size([256, 512])
#     if len(mask) == 1:
#         logp = C * _GCONST_ - 0.5*torch.sum((z+0.5)**2, 1) + logdet_J
#         return logp
#     else:
#         # random_coin = randint(1,10)
#         # if random_coin%2 == 0:
#         #     mean_a = 0.8
#         # else:
#         #     mean_a = 0.8

#         # z_n = torch.empty(z.size())
#         # z_a = torch.empty(z.size())
#         mean_a = 0.5
#         mask_a = mask
#         mask_n = 1 - mask
#         z_a = z*mask_a
#         z_n = z*mask_n
#         logp_n = C * _GCONST_ - 0.5*torch.sum((z_n+0.5)**2, 1)
#         logp_a = C * _GCONST_ - 0.5*torch.sum((z_a-mean_a)**2, 1) / 4.
#         logp = logp_n + logp_a + logdet_J
#         # logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J
#         return logp

# def get_logp(C, z, logdet_J, mask, mean_n, mean_a, var_a):

#     if len(mask) == 1:
#         logp = C * _GCONST_ - 0.5*torch.sum((z-mean_n)**2, 1) + logdet_J
#         return logp
#     else:
#         mask_a = torch.round(mask)
#         mask_n = 1 - mask_a
#         num_a = mask_a.sum()
#         num_n = mask_n.sum()
#         z_a = z*mask_a
#         z_n = z*mask_n
        
#         logp_n = C * _GCONST_ - 0.5*torch.sum((z_n-mean_n)**2, 1)
#         logp_n = logp_n*mask_n.squeeze(1)
#         logp_n = logp_n[logp_n!=0]
        
#         logp_a = C * _GCONST_ - 0.5*torch.sum((z_a-mean_a)**2, 1) / var_a**2
#         logp_a = logp_a*mask_a.squeeze(1)
#         logp_a = logp_a[logp_a!=0]

#         # log_sum = torch.cat((logp_n/num_n, logp_a/num_a))
#         log_sum = torch.cat((logp_n, logp_a))
#         logp = log_sum + logdet_J

#         return logp


# negative log likelihood
def get_logp(C, z, logdet_J, mask):

    if len(mask) == 1:
        logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J
        return logp
    else:
        mask_a = torch.round(mask)
        mask_n = 1 - mask_a
        num_a = mask_a.sum()
        num_n = mask_n.sum()
        z_a = z*mask_a
        z_n = z*mask_n
        
        logp_n = C * _GCONST_ - 0.5*torch.sum(z_n**2, 1)
        logp_n = logp_n*mask_n.squeeze(1)
        # logp_n = logp_n[logp_n!=0]
        
        logp_a = C * _GCONST_ - 0.5*torch.sum(z_a**2, 1)
        logp_a = logp_a*mask_a.squeeze(1)
        # logp_a = logp_a[logp_a!=0]

        # log_sum = torch.cat((logp_n/num_n, logp_a/num_a))
        log_sum = logp_n - logp_a
        # print("log_sum  ", log_sum.size())
        # print("logdet_J ", logdet_J.size())
        logp = log_sum + logdet_J

        return logp


def get_logp_var(C, z, logdet_J, mask, var_n, var_a):
    mean_n = 0
    mean_a = 1
    # print(var.size()) # [256, 1]
    # var_pow = var**2
    # var_pow = var_pow.squeeze(1)
    
    var_ = var_n.squeeze(1)
    # var_1 = var_.detach()

    if len(mask) == 1:
        # logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) / var_ + logdet_J  + C * 0.5 * torch.log(var_)
        # return logp
        std_a = torch.sqrt(var_a) #
        std_n = torch.sqrt(var_).unsqueeze(1) # (256)
        z_nn = (z - mean_n) / std_n # normal to normal
        # z_na = (z - mean_a) / std_a # normal to anomaly, need to be negative
        logp_n = C * _GCONST_ - 0.5*torch.sum(z_nn**2, 1) + logdet_J# - C * 0.5 * torch.log(var_)
        # logp_a = C * _GCONST_ - 0.5*torch.sum(z_na**2, 1) + logdet_J# - C * 0.5 * torch.log(var_)
        # logp = logp_n - logp_a
        return logp_n

    else:
        mask_a = torch.round(mask)
        mask_n = 1 - mask_a
        z_a = z*mask_a # (256, 512)
        z_n = z*mask_n # (256, 512)
        std_a = torch.sqrt(var_a) #
        std_n = torch.sqrt(var_).unsqueeze(1) # (256)
        # print(z_a.size(), z_n.size(), var_a.size(), var_.size(), std_a.size(), std_n.size())
        z_aa1 = (z_a - mean_a) / std_a.detach() # anomaly to anomaly
        z_aa2 = (z_a.detach() - mean_a) / std_a
        z_an1 = (z_a - mean_n) / std_n.detach() # anomaly to normal, need to be negative
        z_an2 = (z_a.detach() - mean_n) / std_n
        z_nn1 = (z_n - mean_n) / std_n.detach() # normal to normal
        z_nn2 = (z_n.detach() - mean_n) / std_n
        z_na1 = (z_n - mean_a) / std_a.detach() # normal to anomaly, need to be negative
        z_na2 = (z_n.detach() - mean_a) / std_a
        # v_n = var_.detach()
        # v_a = var_a.detach()
        
        logp_nn1 = C * _GCONST_ - 0.5*torch.sum(z_nn1**2, 1)
        logp_nn1 = logp_nn1*mask_n.squeeze(1)
        logp_nn2 = C * _GCONST_ - 0.5*torch.sum(z_nn2**2, 1)
        logp_nn2 = logp_nn2*mask_n.squeeze(1)
        logp_nn = logp_nn1 + logp_nn2
        
        logp_an1 = C * _GCONST_ - 0.5*torch.sum(z_an1**2, 1)
        logp_an1 = logp_an1*mask_n.squeeze(1)
        logp_an2 = C * _GCONST_ - 0.5*torch.sum(z_an2**2, 1)
        logp_an2 = logp_an2*mask_n.squeeze(1)
        logp_an = logp_an1 + logp_an2

        logp_aa1 = C * _GCONST_ - 0.5*torch.sum(z_aa1**2, 1)
        logp_aa1 = logp_aa1*mask_a.squeeze(1)
        logp_aa2 = C * _GCONST_ - 0.5*torch.sum(z_aa2**2, 1)
        logp_aa2 = logp_aa2*mask_a.squeeze(1)
        logp_aa = logp_aa1 + logp_aa2
        
        logp_na1 = C * _GCONST_ - 0.5*torch.sum(z_na1**2, 1)
        logp_na1 = logp_na1*mask_a.squeeze(1)
        logp_na2 = C * _GCONST_ - 0.5*torch.sum(z_na2**2, 1)
        logp_na2 = logp_na2*mask_a.squeeze(1)
        logp_na = logp_na1 + logp_na2

        log_sum = logp_nn - logp_an
        # print("log_sum  ", log_sum.size()) # [256]
        # print("logdet_J ", logdet_J.size()) # [256]
        logp = log_sum + logdet_J

        return logp


log_theta = torch.nn.LogSigmoid()

def get_logp_var2(C, z, logdet_J, mask, var_n, var_a):
    mean_n = 0
    mean_a = 1
    
    var_ = var_n.squeeze(1)
    # var_1 = var_.detach()

    if len(mask) == 1:
        # logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) / var_ + logdet_J  + C * 0.5 * torch.log(var_)
        # return logp
        # std_a = torch.sqrt(var_a) #
        std_n = torch.sqrt(var_).unsqueeze(1) # (256)
        z_nn = (z - mean_n) / std_n # normal to normal
        # z_na = (z - mean_a) / std_a # normal to anomaly, need to be negative
        logp_n = C * _GCONST_ - 0.5*torch.sum(z_nn**2, 1) + logdet_J
        # logp_a = C * _GCONST_ - 0.5*torch.sum(z_na**2, 1) + logdet_J# - C * 0.5 * torch.log(var_)
        # logp = logp_n - logp_a
        return logp_n

    else:
        mask_a = torch.round(mask)
        mask_n = 1 - mask_a
        z_a = z*mask_a # (256, 512)
        z_n = z*mask_n # (256, 512)
        
        # logp_nn = C * _GCONST_ - 0.5*torch.sum((z_n-mean_n)**2, 1).detach() / var_ - 0.5*torch.sum((z_n-mean_n)**2, 1) / var_.detach() - C * 0.5 * torch.log(var_)
        # logp_nn = logp_nn*mask_n.squeeze(1)
        # logp_an = C * _GCONST_ - 0.5*torch.sum(1-(z_a-mean_n)**2, 1).detach() / var_a - 0.5*torch.sum(1-(z_a-mean_n)**2, 1) / var_a.detach() - C * 0.5 * torch.log(var_a)
        # logp_an = logp_an*mask_a.squeeze(1)
        # log_sum = logp_nn + logp_an
        
        logp_n =  _GCONST_ - 0.5*(z-mean_n)**2 / var_.unsqueeze(1) - 0.5*torch.log(var_.unsqueeze(1))
        logp_a =  _GCONST_ - 0.5*(z-mean_a)**2 / var_a - 0.5*torch.log(var_a)
        logp_nn = logp_n*mask_n.squeeze(1)
        logp_na = logp_n*mask_n.squeeze(1)
        logp_aa = logp_a*mask_a.squeeze(1)
        logp_an = logp_a*mask_a.squeeze(1)
        
        log_total = (logp_nn - logp_na + logp_aa - logp_an)*1e-3
        outlier = log_theta(log_total)*1e2
        log_sum = logp_nn + logp_aa
        logp = log_sum + logdet_J

        return logp


def get_logp_contrastive(C, z, logdet_J, mask, var_n, var_a):
    mean_n = 0
    mean_a = 1
    var_ = var_n.squeeze(1)
    # var_1 = var_.detach()
    if len(mask) == 1:
        std_n = torch.sqrt(var_).unsqueeze(1) # (256)
        z_nn = (z - mean_n) / std_n # normal to normal
        logp_n = C * _GCONST_ - 0.5*torch.sum(z_nn**2, 1) + logdet_J
        return logp_n

    else:
        mask_a = torch.round(mask)
        mask_n = 1 - mask_a
        z_a = z*mask_a # (256, 512)
        z_n = z*mask_n # (256, 512)
        # a_unit = F.normalize(z.detach(), p=2, dim=0) * math.sqrt(C)

        logp_nn = C * _GCONST_ - 0.5*torch.sum((z-mean_n)**2, 1) / var_ - 0.5 * torch.log(var_)
        # logp_nn = logp_nn*mask_n.squeeze(1)
        # logp_n = logp_nn

        logp_aa = C * _GCONST_ - 0.5*torch.abs(torch.sum(z_a**2, 1) - C) / var_a - 0.5 * torch.log(var_a)
        # logp_aa = C * _GCONST_ - 0.5*torch.sum((z-a_unit)**2, 1) / var_a - 0.5 * torch.log(var_a)
        # logp_aa = logp_aa*mask_a.squeeze(1)
        # logp_a = logp_aa
        
        log_sum_n = (logp_nn + logdet_J) * mask_n.squeeze(1) #- (logp_nn + logdet_J) * mask_a.squeeze(1)
        log_sum_a = (logp_aa + logdet_J) * mask_a.squeeze(1)
        logp = log_sum_n + log_sum_a

        return logp


def get_logp_gmm(C, z, logdet_J, mask, var_n, var_a):
    mean_n = 0
    mean_a1 = 1
    mean_a2 = -1
    var_ = var_n.squeeze(1)
    if len(mask) == 1:
        std_n = torch.sqrt(var_).unsqueeze(1) # (256)
        z_nn = (z - mean_n) / std_n # normal to normal
        logp_n = C * _GCONST_ - 0.5*torch.sum(z_nn**2, 1) + logdet_J
        return logp_n

    else:
        mask_a = torch.round(mask)
        mask_n = 1 - mask_a
        z_a = z*mask_a # (256, 512)
        z_n = z*mask_n # (256, 512)
        
        za1 = z_a - mean_a1
        za2 = z_a - mean_a2
        stacked = torch.stack([torch.sum(za1**2, 1), torch.sum(za2**2, 1)], dim=0)
        za = torch.amin(stacked, dim=0)

        logp_nn = C * _GCONST_ - 0.5*torch.sum((z-mean_n)**2, 1) / var_ - 0.5 * torch.log(var_)
        # logp_nn = logp_nn*mask_n.squeeze(1)
        # logp_n = logp_nn
        # logp_na = (logp_nn + logdet_J)*mask_a.squeeze(1)

        # logp_aa = C * _GCONST_ - 0.5*torch.sum((z_a-mean_a)**2, 1).detach() / var_a - 0.5*torch.sum((z_a-mean_a)**2, 1) / var_a.detach() - C * 0.5 * torch.log(var_a)
        logp_aa = C * _GCONST_ - 0.5*za.detach() / var_a - 0.5*za / var_a.detach() - 0.5 * torch.log(var_a)
        logp_aa = logp_aa*mask_a.squeeze(1)
        logp_a = logp_aa
        
        log_sum = logp_nn + logp_aa 
        logp = log_sum + logdet_J #- logp_na

        return logp


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())
