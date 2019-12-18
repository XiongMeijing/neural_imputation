import torch.distributions as td
import torch
import torch.nn.functional as F


class DistributionMixin(td.Distribution):
    """
    The distributions here are mostly just wrappers for Torch distributions, but a some
    tweaks to make them do thinks like instantiate under new values easier.
    """

    @property
    def param_size(self):
        '''
        The number of parameters from the neural network output that are needed
        to create an instance of the distribution in pytorch
        '''
        raise NotImplementedError

    @property
    def required_params(self):
        '''
        Names of the required parameters, if any
        '''
        raise NotImplementedError

    def independent(self, reinterpreted_batch_ndims=1):
        '''
        Flattening the data into one (or more) dimensions and using as if it were
        td.Independent(distribution=OurDistribution...) is common
        '''
        return td.Independent(self, reinterpreted_batch_ndims=reinterpreted_batch_ndims)

    def sample(self, sample_shape):
        '''
        This is a more robust call to the torch distribution's sampling, using the
        reparameterized version where possible
        '''
        return self.rsample(sample_shape) if self.has_rsample else self.sample(sample_shape)


class Normal(td.Normal, DistributionMixin):
    param_size = 2
    required_params = ["loc", "scale"]

    def __init__(self):
        pass

    def create(self, params):
        if len(params) != self.param_size:
            raise ValueError('The number of parameters in the list is not correct')

        loc, scale_ = td.utils.broadcast_all(params[0], params[1])
        scale = F.softplus(scale_) + 1e-6

        super(Normal, self).__init__(loc=loc, scale=scale)


class Laplace(td.Laplace, DistributionMixin):
    param_size = 2
    required_params = ["loc", "scale"]

    def __init__(self):
        pass

    def create(self, params):
        if len(params) != self.param_size:
            raise ValueError('The number of parameters in the list is not correct')

        loc, scale_ = td.utils.broadcast_all(params[0], params[1])
        scale = F.softplus(scale_) + 1e-6

        super(Laplace, self).__init__(loc=loc, scale=scale)

class StudentT(td.StudentT, DistributionMixin):
    param_size = 3
    required_params = ["loc", "scale", "df"]

    def __init__(self):
        pass

    def create(self, params):
        if len(params) != self.param_size:
            raise ValueError('The number of parameters in the list is not correct')

        loc, scale_, df_ = td.utils.broadcast_all(params[0], params[1], params[2])
        scale = F.softplus(scale_) + 1e-6
        df = df_ + 3

        super(StudentT, self).__init__(loc=loc, scale=scale, df=df)

class Bernoulli(td.Bernoulli, DistributionMixin):
    param_size = 1
    required_params = ["logits"]

    def __init__(self):
        pass

    def create(self, params):
        if len(params) != self.param_size:
            raise ValueError('The number of parameters in the list is not correct')

        logits = params[0]

        super(Bernoulli, self).__init__(logits=logits)