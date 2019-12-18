import torch.nn.functional as F
import torch

def betavae_loss(distortion, rate, beta=1.0):
    elbo_betavae = beta * vae_loss(betavae_loss)
    return elbo_betavae

def vae_loss(distortion, rate):
    elbo_local = distortion+rate
    elbo_vae = -torch.mean(elbo_local.view(-1,1),0)
    return elbo_vae

def iwae_loss(distortion, rate):
    K = distortion.shape(0)
    elbo_local = distortion+rate
    elbo_iwae = -torch.mean(torch.logsumexp(elbo_local,0)) + torch.log(torch.Tensor([K]))
    return elbo_iwae

