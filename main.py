# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import ipdb

import data
from model import BBBRNNModel

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/ptb/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=2,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')

#BBB Config
parser.add_argument('--BBB', action='store_true', help='use BBB')
parser.add_argument('--sharpen', action='store_true', help='use posterior sharpening')
parser.add_argument('--pi', type=float, default=0.25)
parser.add_argument('--logstd1', type=float, default=0)
parser.add_argument('--logstd2', type=float, default=-6)

# log
parser.add_argument('--logdir', default='./logs/')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
if args.seed is not None:
    torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        if args.seed is not None:
            torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = BBBRNNModel(
    args.model, args.sharpen, ntokens, args.emsize, args.nhid,
    args.nlayers, args.dropout, args.tied,
    pi=args.pi, logstd1=args.logstd1,
    logstd2=args.logstd2, BBB=args.BBB,
    gpu=args.cuda
)


if args.cuda:
    model.cuda()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_nll = 0
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0)- 1, args.bptt):
    #for i in range(0, 100, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden, targets)
        NLL = model.get_nll(output, targets).data
        total_nll += len(data) * NLL
        hidden = repackage_hidden(hidden)
    return total_nll[0] / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()

    avg_loss = 0
    avg_nll = 0
    avg_kl = 0

    total_loss = 0
    total_nll = 0
    total_kl = 0

    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    num_batch = train_data.size(0) / args.bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
    #for batch, i in enumerate(range(0, 100, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden, targets)
        NLL, KL, KL_sharp = model.get_loss(output, targets)

        # proper scaling for a batch loss
        NLL_term = NLL * args.bptt # \frac{1}{C} \sum_{c=1}^C p(y^c|x^c)
        KL_term = KL / (num_batch * args.batch_size) # KL(q|p) / BC
        loss = NLL_term + KL_term
        if args.sharpen:
            loss += KL_sharp / num_batch
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data
        total_nll += NLL.data
        total_kl += KL.data

        avg_loss += loss.data / num_batch
        avg_nll += NLL.data / num_batch
        avg_kl += KL.data / num_batch

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            curr_nll = total_nll[0] / args.log_interval
            curr_kl = total_kl[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | kl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(curr_nll), curr_kl))
            total_loss = 0
            total_kl = 0
            total_nll = 0
            start_time = time.time()

    return avg_loss[0], avg_nll[0], avg_kl[0]

# Loop over epochs.
lr = args.lr
best_val_loss = None

# prepare logdir
if not os.path.isdir(args.logdir):
    os.mkdir(args.logdir)
model_path = os.path.join(args.logdir, "model.pt")

train_losses = []
train_kls = []

train_ppls = []
val_ppls = []

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()

        train_loss, train_nll, train_kl = train()
        train_losses.append(train_loss)
        train_ppls.append(math.exp(train_nll))
        train_kls.append(train_kl)

        val_loss = evaluate(val_data)
        val_ppls.append(math.exp(val_loss))
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(model_path, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


# save plots
plot_dir = os.path.join(args.logdir, "plots")
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)
plt.figure()
plt.plot(train_losses)
plt.xlabel("epochs")
plt.ylabel("train loss")
plt.savefig(os.path.join(plot_dir, "train_loss.png"))
plt.close()

plt.figure()
plt.plot(train_kls)
plt.xlabel("epochs")
plt.ylabel("train KL")
plt.savefig(os.path.join(plot_dir, "train_kl.png"))
plt.close()

plt.figure()
train_ppls_plt, = plt.plot(train_ppls)
val_ppls_plt, = plt.plot(val_ppls)
plt.legend([train_ppls_plt, val_ppls_plt], ["train ppl", "val ppl"])
plt.xlabel("epochs")
plt.ylabel("Perplexity")
plt.savefig(os.path.join(plot_dir, "ppl.png"))
plt.close()

# save param
with open(os.path.join(args.logdir, "param.txt"), 'w') as f:
    args_dict = args.__dict__
    for k in args_dict.keys():
        f.write("{} {}\n".format(k, args_dict[k]))

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
