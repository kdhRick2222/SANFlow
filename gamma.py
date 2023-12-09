import torch
import torch.nn as nn
from torch.autograd import Function as autoF
from scipy.special import gammaln
import numpy as np

class LogGamma(autoF):
    '''
    Implement of the logarithm of gamma Function.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if input.is_cuda:
            input_np = input.detach().cpu().numpy()
        else:
            input_np = input.detach().numpy()
        out = gammaln(input_np)
        out = torch.from_numpy(out).to(device=input.device).type(dtype=input.dtype)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.digamma(input) * grad_output

        return grad_input