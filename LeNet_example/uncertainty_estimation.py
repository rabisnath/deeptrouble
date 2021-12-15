'''
Credit to Kumar Shridhar
https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/uncertainty_estimation.py
'''

import torch
from torch.nn import functional as F
import numpy as np

def get_uncertainty_per_image(model, input_image, T=15, normalized=False):
    
    '''
    original code:
    
    input_image = input_image.unsqueeze(0)
    input_images = input_image.repeat(T, 1, 1, 1)
    net_out, _ = model(input_images)
    '''

    #net_out = [model(img) for img in input_images]

    net_out = model(input_image)

    pred = torch.mean(net_out, dim=0).cpu().detach().numpy()
    if normalized:
        prediction = F.softplus(net_out)
        p_hat = prediction / torch.sum(prediction, dim=1).unsqueeze(1)
    else:
        p_hat = F.softmax(net_out, dim=1)
    p_hat = p_hat.detach().cpu().numpy()
    p_bar = np.mean(p_hat, axis=0)

    temp = p_hat - np.expand_dims(p_bar, 0)
    epistemic = np.dot(temp.T, temp) / T
    epistemic = np.diag(epistemic)

    aleatoric = np.diag(p_bar) - (np.dot(p_hat.T, p_hat) / T)
    aleatoric = np.diag(aleatoric)

    return pred, epistemic, aleatoric