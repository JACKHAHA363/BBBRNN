import torch.nn as nn
import torch
import math
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence

def mul_var_normal(weights, means, logvars):
    """
    theta is from a multivariate gaussian with diagnol covariance
    return the loglikelihood.
    :param weights: a list of weights
    :param means: a list of means
    :param logvars: a list of logvars
    :return ll: loglikelihood sum over list
    """
    ll = 0

    for i in range(len(weights)):
        w = weights[i]
        mean = means[i]
        if len(logvars) > 1:
            logvar = logvars[i]
            var = logvar.exp()
        else:
            logvar = logvars[0]
            var = math.exp(logvar)

        logstd = logvar * 0.5
        ll += torch.sum(
            -((w - mean)**2)/(2*var) - logstd - math.log(math.sqrt(2*math.pi))
        )

    return ll

def gaussian_mix(weights, pi, logstd1, logstd2):
    """
    :param weights: a list of weights
    :param pi: number
    :param logstd1: number
    :param logstd2: number
    :return ll: likelihood sum over the list
    """
    ll = 0
    for w in weights:
        var1 = math.exp(logstd1 * 2)
        ll1 = torch.sum(
            -(w**2)/(2*var1) - logstd1 - math.log(math.sqrt(2*math.pi))
        )


        var2 = math.exp(logstd2 * 2)
        ll2 = torch.sum(
            -(w**2)/(2*var2) - logstd2 - math.log(math.sqrt(2*math.pi))
        )

        # use a numerical stable one
        # ll1 + log(pi + (1-pi) exp(ll2-ll1))
        ll += ll1 + ( pi + (1-pi) * ((ll2-ll1).exp()) ).log()

    return ll

class BBBLayer(nn.Module):
    """
    a base class for all BBB layer with gaussian mixture prior
    """
    def __init__(self, pi, logstd1, logstd2, gpu, BBB):
        super(BBBLayer, self).__init__()
        self.pi = pi
        self.logstd1 = logstd1
        self.logstd2 = logstd2
        self.gpu = gpu
        self.BBB = BBB

        self.sampled_weights = []
        self.sampled_sharpen_weights = []
        self.means = []
        self.logvars = []
        self.h_post_means = []

    def sample(self):
        assert self.BBB is True
        self.sampled_weights = [] # clear samples
        for i in range(len(self.means)):
            mean = self.means[i]
            logvar = self.logvars[i]
            eps = torch.zeros(mean.size())
            if self.gpu:
                eps = eps.cuda()

            eps.normal_()
            std = logvar.mul(0.5).exp()
            weight = mean + Variable(eps) * std
            self.sampled_weights.append(weight)

    def resample_with_sharpening(self, grads, eta, std = 0.02):
        self.sampled_sharpen_weights = []
        self.h_post_means = []
        for i in range(len(self.sampled_weights)):
            w = self.sampled_weights[i]
            # Random number
            eps = torch.zeros(w.size())
            if self.gpu:
                eps = eps.cuda()

            eps.normal_()
            g = grads[i].detach()
            # Sample fron normal wih posterior sharpening
            h_post_means = (w - eta[i] * g)
            weight = h_post_means + Variable(eps) * std
            self.h_post_means.append(h_post_means)
            self.sampled_sharpen_weights.append(weight)

    def get_kl_sharpening(self, sigma=0.02):
        kl = 0
        for i in range(len(self.sampled_weights)):
            sharp_w = self.sampled_sharpen_weights[i]
            w = self.sampled_weights[i].detach()

            # without constant term
            kl += torch.sum((sharp_w - w).pow(2) / (2*sigma**2))

        return kl

    def get_kl(self):
        """
        Use the current sampled weights to calculate the KL divergence from posterior to prior.
        :return: The KL
        """
        assert len(self.sampled_weights) != 0 # make sure we sample weights

        log_posterior = mul_var_normal(
            weights=self.sampled_weights,
            means=[ mean.detach() for mean in self.means],
            logvars=[ logvar.detach() for logvar in self.logvars]
        )
        log_prior = gaussian_mix(self.sampled_weights, pi=self.pi, logstd1=self.logstd1, logstd2=self.logstd2)
        kl = log_posterior - log_prior
        return kl

class BBBLinear(BBBLayer):
    """
    adapted from torch.nn.Linear
    with Gaussian mixture as prior
    """
    def __init__(self, in_features, out_features, *args, **kwargs):
        super(BBBLinear, self).__init__(*args, **kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.weight_mean = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mean = nn.Parameter(torch.Tensor(out_features))
        if self.BBB is True:
            self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias_logvar = nn.Parameter(torch.Tensor(out_features))

        # used for KL
        self.means = [self.weight_mean, self.bias_mean]
        if self.BBB is True:
            self.logvars = [self.weight_logvar, self.bias_logvar]

        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_mean.size(1))
        logvar_init = math.log(stdv) * 2
        for mean in self.means:
            mean.data.uniform_(-stdv, stdv)
        if self.BBB is True:
            for logvar in self.logvars:
                logvar.data.fill_(logvar_init)

    def forward(self, inputs):
        if self.training and self.BBB is True:
            # if use BBB and it is training
            self.sample()
            weight = self.sampled_weights[0]
            bias = self.sampled_weights[1]
        else:
            # use only mean for testing or non BBB
            weight = self.weight_mean
            bias = self.bias_mean
        return nn.functional.linear(inputs, weight, bias)


class BBBRNN(BBBLayer):

    def __init__(self, mode, sharpen, input_size, hidden_size,
                 num_layers=1, batch_first=False,
                 dropout=0, bidirectional=False, *args, **kwargs):
        super(BBBRNN, self).__init__(*args, **kwargs)
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        self.num_directions = num_directions
        self.smoohing = sharpen

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        else:
            gate_size = hidden_size

        self.means = []
        self.logvars = []
        self.eta = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih_mean = nn.Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh_mean = nn.Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih_mean = nn.Parameter(torch.Tensor(gate_size))
                b_hh_mean = nn.Parameter(torch.Tensor(gate_size))
                self.means += [w_ih_mean, w_hh_mean, b_ih_mean, b_hh_mean]

                if self.BBB is True:
                    w_ih_logvar = nn.Parameter(torch.Tensor(gate_size, layer_input_size))
                    w_hh_logvar = nn.Parameter(torch.Tensor(gate_size, hidden_size))
                    b_ih_logvar = nn.Parameter(torch.Tensor(gate_size))
                    b_hh_logvar = nn.Parameter(torch.Tensor(gate_size))
                    self.logvars += [w_ih_logvar, w_hh_logvar, b_ih_logvar, b_hh_logvar]

                # set weight to be attribute
                if self.BBB is True:
                    layer_params = (
                        w_ih_mean, w_ih_logvar,
                        w_hh_mean, w_hh_logvar,
                        b_ih_mean, b_ih_logvar,
                        b_hh_mean, b_hh_logvar
                    )
                    suffix = '_reverse' if direction == 1 else ''
                    param_names = ['weight_ih_mean_l{}{}', 'weight_ih_logvar_l{}{}', 'weight_hh_mean_l{}{}', 'weight_hh_logvar_l{}{}']
                    param_names += ['bias_ih_mean_l{}{}',  'bias_ih_logvar_l{}{}', 'bias_hh_mean_l{}{}',  'bias_hh_logvar_l{}{}']
                else:
                    layer_params = (
                            w_ih_mean,
                            w_hh_mean,
                            b_ih_mean,
                            b_hh_mean
                            )
                    suffix = '_reverse' if direction == 1 else ''
                    param_names = ['weight_ih_mean_l{}{}', 'weight_hh_mean_l{}{}']
                    param_names += ['bias_ih_mean_l{}{}',  'bias_hh_mean_l{}{}']

                param_names = [x.format(layer, suffix) for x in param_names]
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)

                if self.smoohing:
                    w_ih_eta = nn.Parameter(torch.Tensor(gate_size, layer_input_size))
                    w_hh_eta = nn.Parameter(torch.Tensor(gate_size, hidden_size))
                    b_ih_eta = nn.Parameter(torch.Tensor(gate_size))
                    b_hh_eta = nn.Parameter(torch.Tensor(gate_size))
                    self.eta += [w_ih_eta, w_hh_eta, b_ih_eta, b_hh_eta]
                    layer_params_sharpen = (
                            w_ih_eta,
                            w_hh_eta,
                            b_ih_eta,
                            b_hh_eta
                            )
                    suffix = '_reverse' if direction == 1 else ''
                    param_names = ['weight_ih_eta_l{}{}', 'weight_hh_eta_l{}{}']
                    param_names += ['bias_ih_eta_l{}{}',  'bias_hh_eta_l{}{}']
                    param_names = [x.format(layer, suffix) for x in param_names]
                    for name, param in zip(param_names, layer_params_sharpen):
                        setattr(self, name, param)

        self.reset_parameters()


    def reset_parameters(self):
        """
        init parameters
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        logvar_init = math.log(stdv) * 2
        for mean in self.means:
            mean.data.uniform_(-stdv, stdv)
        if self.BBB is True:
            for logvar in self.logvars:
                logvar.data.fill_(logvar_init)

            if self.smoohing:
                for eta in self.eta:
                    eta.data.uniform_(-stdv, stdv)

    def _apply(self, fn):
        ret = super(BBBRNN, self)._apply(fn)
        return ret

    def get_all_weights(self, weights):
        """
        a helper function that transform a list of weights
        to pytorch RNN backend weight
        """
        start = 0
        all_weights = []
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                w_ih = weights[start]
                w_hh = weights[start+1]
                b_ih = weights[start+2]
                b_hh = weights[start+3]
                start += 4
                all_weights.append([w_ih, w_hh, b_ih, b_hh])

        return all_weights

    def forward(self, input, hx=None, grads=None):
        if grads is not None:
            self.resample_with_sharpening(grads, self.eta)
            weights = self.sampled_sharpen_weights
        elif self.training and self.BBB is True:
            self.sample()
            weights = self.sampled_weights
        else:
            weights = self.means

        # modify weights to pytorch format
        self.all_weights = self.get_all_weights(weights)
        # RNN base code
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.autograd.Variable(input.data.new(self.num_layers *
                                                        num_directions,
                                                        max_batch_size,
                                                        self.hidden_size).zero_(), requires_grad=False)
            if self.mode == 'LSTM':
                hx = (hx, hx)

        func = self._backend.RNN(
            self.mode,
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            batch_sizes=batch_sizes,
            dropout_state=self.dropout_state,
            flat_weight=None
        )
        # change this line
        output, hidden = func(input, self.all_weights, hx)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden
