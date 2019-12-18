import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import neural_imputation.utils.distributions as nid
import numpy as np

from neural_imputation.utils.distributions import Normal, Laplace, StudentT, Bernoulli
from neural_imputation.utils import metrics
from neural_imputation.utils.autoencoder import Encoder, Decoder

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


DISTRIBUTIONS = ['normal', 'gaussian','laplace','studentt','bernoulli']


class VAE(nn.Module):

    def __init__(self, q_distribution, p_distribution, z_prior, input_size, code_size,
                 encoder_hidden_sizes=[], decoder_hidden_sizes=[],
                 K_train=1, K_test=1, variable_sizes=None):
        super(VAE, self).__init__()

        # Set the inditial distribution classes
        for dist_arg, dist in zip([z_prior, q_distribution, p_distribution],
                                  [self.z_prior,self.q_distribution,self.p_distribution]):
            assert dist_arg.lower() in DISTRIBUTIONS, "Not a supported distribution"
            if dist_arg.lower() in ['normal','gaussian']:
                dist = nid.Normal()
            elif dist_arg.lower() == 'laplace':
                dist = nid.Laplace()
            elif z_prior.lower() == 'studentt':
                dist = nid.StudentT()
            
        self.input_size = input_size
        self.code_size = code_size
        self.K_train = K_train
        self.K_test = K_test
        
        self.encoder = Encoder(input_size, self.q_distribution.param_size * code_size, hidden_sizes=encoder_hidden_sizes)
        self.decoder = Decoder(code_size, self.p_distribution.param_size * input_size, hidden_sizes=decoder_hidden_sizes)

    def forward(self, inputs, training=True):
        d = self.code_size
        K = self.K_train if training else self.K_test

        batch_data, batch_mask = inputs
        mb = batch_data.shape[0]
        data_tiled = torch.Tensor.repeat(batch_data, [K, 1])
        mask_tiled = torch.Tensor.repeat(batch_mask, [K, 1])
        
        #create prior
        if self.z_prior.param_size == 2: #loc, scale
            softmaxone = np.log(2.718281-1)
            prior_shape = [mb,d]
            self.z_prior.create([torch.zeros(prior_shape),torch.ones(prior_shape) * softmaxone])
        elif self.z_prior.param_size == 3: #loc, scale, df
            softmaxone = np.log(2.718281-1)
            prior_shape = [mb,d]
            self.z_prior.create([torch.zeros(prior_shape),torch.ones(prior_shape) * softmaxone,
                                 torch.ones(prior_shape) * 5.])

        self.encode(batch_data, K = K)
        z_sampled = self.q_distribution.independent().sample([K]) # shape: [K, mb, d]
        logpz = self.z_prior.independent().log_prob(z_sampled) # shape: [K, mb]
        logqz = self.q_distribution.independent().log_prob(z_sampled) # shape: [K, mb]
        rate_i = logqz - logpz # shape: [K, mb]


        self.decode(z_sampled.view([K * mb, d]), training=training)  # creates the p_distribution
        distortion_i = -self.p_distribution.log_prob(data_tiled) # logp_givenz is the neg. loglikelihood = -distortion
        distortion_obs_i = torch.sum(distortion_i * mask_tiled, 1).view([K, mb])


        return distortion_obs_i, rate_i

    def encode(self, inputs):
        d = self.code_size

        outputs = self.encoder(inputs)
        params = [outputs[..., (i * d):((i + 1) * d)] for i in range(self.q_distribution.param_size)]
        self.q_distribution.create(params)

        return self

    def decode(self, inputs, training=False):
        p = self.input_size

        outputs = self.decoder(inputs)
        params = [outputs[..., (i * p):((i + 1) * p)] for i in range(self.p_distribution.param_size)]
        self.p_distribution.create(params)

        return self

    def sample(self, inputs):
        with torch.no_grad():
            K = self.K_test
            rate, distortion = self.forward(inputs, training=False)
            imp_weights = F.softmax(distortion + rate, 0)  # these are w_1,....,w_L for all observations in the batch
            x_sample = self.p_distribution.independent().sample()
            xms = x_sample.view([K, -1, self.input_size])
            xm = torch.einsum('ki,kij->ij', imp_weights, xms)
            return xm