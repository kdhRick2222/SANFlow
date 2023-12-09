import os, math
import numpy as np
import torch

RESULT_DIR = './results'
WEIGHT_DIR = './weights'
WEIGHT_DIR_FF = './weights_ff'
RESULT_DIR_FF = './results_ff'
RESULT_DIR_v = './results_var'
WEIGHT_DIR_v = './weights_var'
RESULT_DIR_concat = './results_concat'
WEIGHT_DIR_concat = './weights_concat'
RESULT_DIR_m = './results_mean_likelihood'
WEIGHT_DIR_m = './weights_mean_likelihood'
MODEL_DIR  = './models'

__all__ = ('save_weights_var', 'load_weights_var', 'save_all', 'adjust_learning_rate', 'warmup_learning_rate')

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def save_all(auc_all, class_name, run_date):
    l = len(auc_all)
    result = ''
    for i in range(l):
        # result += '{:d} \t {:.2f} \t {:.2f}\n'.format(auc_all[i][0], auc_all[i][1], auc_all[i][2])
        result += '{:d} \t {:.2f} \t {:.2f} \t {:.2f}\n'.format(auc_all[i][0], auc_all[i][1], auc_all[i][2], auc_all[i][3])
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    fp = open(os.path.join(RESULT_DIR, '{}_{}.txt'.format(class_name, run_date)), "w")
    fp.write(result)
    fp.close()


def save_weights_var(encoder, decoders, model_name, run_date, var):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    state = {'encoder_state_dict': encoder.state_dict(),
             'decoder_state_dict': [decoder.state_dict() for decoder in decoders],
             'variance': var.state_dict()}
    filename = '{}_{}.pt'.format(model_name, run_date)
    path = os.path.join(WEIGHT_DIR, filename)
    torch.save(state, path)
    print('Saving weights to {}'.format(filename))


def load_weights_var(encoder, decoders, var, filename):
    path = os.path.join(filename)
    state = torch.load(path)
    encoder.load_state_dict(state['encoder_state_dict'], strict=False)
    decoders = [decoder.load_state_dict(state, strict=False) for decoder, state in zip(decoders, state['decoder_state_dict'])]
    var.load_state_dict(state['variance'], strict=False)
    print('Loading weights from {}'.format(filename))


def adjust_learning_rate(c, optimizer, epoch):
    lr = c.lr
    if c.lr_cosine:
        eta_min = lr * (c.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / c.meta_epochs)) / 2
    else:
        steps = np.sum(epoch >= np.asarray(c.lr_decay_epochs))
        if steps > 0:
            lr = lr * (c.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(c, epoch, batch_id, total_batches, optimizer):
    if c.lr_warm and epoch < c.lr_warm_epochs:
        p = (batch_id + epoch * total_batches) / \
            (c.lr_warm_epochs * total_batches)
        lr = c.lr_warmup_from + p * (c.lr_warmup_to - c.lr_warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #
    for param_group in optimizer.param_groups:
        lrate = param_group['lr']
    return lrate

