import torch.nn as nn
from BBBLayers import BBBLinear
from BBBLayers import BBBRNN
from torch.autograd import Variable
import torch

class BBBRNNModel(nn.Module):
    """
    Modify from language model pytorch exampl
    """
    def __init__(
            self, rnn_type, sharpen, ntoken, ninp,
            nhid, nlayers, dropout=0.5,
            tie_weights=False, *args, **kwargs
    ):
        super(BBBRNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.sharpen = sharpen
        self.rnn = BBBRNN(
            rnn_type, sharpen, ninp, nhid, nlayers, dropout=dropout,
            *args, **kwargs
        )

        self.decoder = BBBLinear(nhid, ntoken, *args, **kwargs)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight_mean = self.encoder.weight

        # init embedding
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntoken = ntoken

        self.layers = [self.rnn, self.decoder]
        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, input, hidden, targets):
        """
        :param input: [seq_len, bsz, inp_dim]
        :return: [seq_len, bsz, inp_dim]
        """
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0)*output.size(1), output.size(2))
        )
        outputs = decoded.view(output.size(0), output.size(1), decoded.size(1))
        if self.sharpen and self.training:
            # We compute the cost
            NLL = self.get_nll(outputs, targets)
            # The gradients
            gradients = torch.autograd.grad(outputs=NLL, inputs=self.rnn.sampled_weights, grad_outputs=torch.ones(NLL.size()).cuda(), create_graph=True, retain_graph=True, only_inputs=True)
            # Then we do the forward pass again with sharpening:
            output, hidden = self.rnn(emb, hidden, gradients)
            decoded = self.decoder(
                    output.view(output.size(0)*output.size(1), output.size(2))
                    )
            outputs = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return outputs, hidden


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def get_nll(self, output, targets):
        # \sum log P(batch | theta) / (bsz * seq_len)
        return self.loss_fn(output.view(-1, self.ntoken), targets)

    def get_loss(self, output, targets):
        """
        return:
            NLL: NLL is averaged over seq_len and batch_size
            KL: KL is the original scale KL
        """
        # NLL
        NLL = self.get_nll(output, targets)

        # KL
        KL = torch.zeros(1)
        if self.rnn.gpu:
            KL = KL.cuda()
        KL = Variable(KL)

        for layer in self.layers:
            if layer.BBB:
                KL += layer.get_kl()

        if self.sharpen:
            KL_sharp = self.rnn.get_kl_sharpening()
        else:
            KL_sharp = 0.
        return NLL, KL, KL_sharp
