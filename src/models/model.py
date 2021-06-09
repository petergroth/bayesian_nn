import pyro
import torch
import pyro.distributions as dist
from torch import nn
from pyro.distributions import constraints
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.contrib import bnn
from pyro.infer import Predictive

class BNN(PyroModule):
    def __init__(self, hidden_size, n_classes):
        super(BNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_input = 784

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.n_input, out_features=self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.n_classes),
            nn.LogSoftmax(dim=0)
        )

        pyro.nn.module.to_pyro_module_(self.fc)

        for m in self.fc.children():
            for name, value in list(m.named_parameters(recurse=False)):
                setattr(m, name, PyroSample(prior=dist.Normal(0., 1.)
                                            .expand(value.shape)
                                            .to_event(value.dim())))

    def forward(self, x: torch.Tensor, obs=None):
        x = x.view(-1, 784)
        logits = self.fc(x)
        with pyro.plate('obs', len(x)):
            y = pyro.sample('y', dist.OneHotCategorical(logits=logits), obs=obs)
        return y

    def guide(self, x: torch.Tensor, obs=None):

        fc1w_mu = torch.randn_like(self.fc[0].weight)
        fc1w_mu_param = pyro.param('0w_mu', fc1w_mu)
        fc1w_sigma = torch.ones_like(self.fc[0].weight)
        fc1w_sigma_param = pyro.param('0w_sigma', fc1w_sigma, constraint=constraints.positive)
        fc1w_prior = pyro.sample('0.weight', dist.Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param).to_event())

        fc1b_mu = torch.randn_like(self.fc[0].bias)
        fc1b_mu_param = pyro.param('0b_mu', fc1b_mu)
        fc1b_sigma = torch.ones_like(self.fc[0].bias)
        fc1b_sigma_param = pyro.param('0b_sigma', fc1b_sigma, constraint=constraints.positive)
        fc1b_prior = pyro.sample('0.bias', dist.Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param).to_event())

        fc2w_mu = torch.randn_like(self.fc[2].weight)
        fc2w_mu_param = pyro.param('2w_mu', fc2w_mu)
        fc2w_sigma = torch.ones_like(self.fc[2].weight)
        fc2w_sigma_param = pyro.param('2w_sigma', fc2w_sigma, constraint=constraints.positive)
        fc2w_prior = pyro.sample('2.weight', dist.Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param).to_event())

        fc2b_mu = torch.randn_like(self.fc[2].bias)
        fc2b_mu_param = pyro.param('2b_mu', fc2b_mu)
        fc2b_sigma = torch.ones_like(self.fc[2].bias)
        fc2b_sigma_param = pyro.param('2b_sigma', fc2b_sigma, constraint=constraints.positive)
        fc2b_prior = pyro.sample('2.bias', dist.Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param).to_event())

        fc3w_mu = torch.randn_like(self.fc[4].weight)
        fc3w_mu_param = pyro.param('3w_mu', fc3w_mu)
        fc3w_sigma = torch.ones_like(self.fc[4].weight)
        fc3w_sigma_param = pyro.param('3w_sigma', fc3w_sigma, constraint=constraints.positive)
        fc3w_prior = pyro.sample('4.weight', dist.Normal(loc=fc3w_mu_param, scale=fc3w_sigma_param).to_event())

        fc3b_mu = torch.randn_like(self.fc[4].bias)
        fc3b_mu_param = pyro.param('3b_mu', fc3b_mu)
        fc3b_sigma = torch.ones_like(self.fc[4].bias)
        fc3b_sigma_param = pyro.param('3b_sigma', fc3b_sigma, constraint=constraints.positive)
        fc3b_prior = pyro.sample('4.bias', dist.Normal(loc=fc3b_mu_param, scale=fc3b_sigma_param).to_event())
