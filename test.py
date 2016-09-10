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
parser.add_argument('--logdir', default='./logs/')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--bptt', default=35)

args = parser.parse_args()

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

eval_batch_size = 50
train_data = batchify(corpus.train, eval_batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

train_rev_data = batchify(corpus.train_rev, eval_batch_size)
val_rev_data = batchify(corpus.valid_rev, eval_batch_size)
test_rev_data = batchify(corpus.test_rev, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)

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
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden, targets)
        NLL = model.get_nll(output, targets).data
        total_nll += len(data) * NLL
        hidden = repackage_hidden(hidden)
    return total_nll[0] / len(data_source)

# Load the best saved model.
model_path = os.path.join(args.logdir, "./model.pt")
with open(model_path, 'rb') as f:
    model = torch.load(f)

if args.cuda:
    model.cuda()

# Run on test data.
train_loss = evaluate(train_data)
train_rev_loss = evaluate(train_rev_data)
print('=' * 89)
print('train ppl {:8.2f}, train_rev ppl {:8.2f}'.format
        (math.exp(train_loss), math.exp(train_rev_loss))
)
print('=' * 89)

val_loss = evaluate(val_data)
val_rev_loss = evaluate(val_rev_data)
print('=' * 89)
print('val ppl {:8.2f}, val_rev ppl {:8.2f}'.format
        (math.exp(val_loss), math.exp(val_rev_loss))
)
print('=' * 89)

test_loss = evaluate(test_data)
test_rev_loss = evaluate(test_rev_data)
print('=' * 89)
print('test ppl {:8.2f}, test_rev ppl {:8.2f}'.format
        (math.exp(test_loss), math.exp(test_rev_loss))
)
print('=' * 89)

