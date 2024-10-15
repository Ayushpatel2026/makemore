import torch
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()
# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)


# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
  X, Y = [], []
  
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%


# utility function we will use later when comparing manual gradients to PyTorch gradients
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 64 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
# Layer 1
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.1 # using b1 just for fun, it's useless because of BN
# Layer 2
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

# Note: I am initializating many of these parameters in non-standard ways
# because sometimes initializating with e.g. all zeros could mask an incorrect
# implementation of the backward pass.

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
#print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True


batch_size = 32
n = batch_size # a shorter variable also, for convenience

for i in range(1):
    # construct a minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
    # forward pass, "chunkated" into smaller steps that are possible to backward one at a time

    emb = C[Xb] # embed the characters into vectors
    embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
    # Linear layer 1
    hprebn = embcat @ W1 + b1 # hidden layer pre-activation
    # BatchNorm layer
    bnmeani = 1/n*hprebn.sum(0, keepdim=True)
    bndiff = hprebn - bnmeani
    bndiff2 = bndiff**2
    bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
    bnvar_inv = (bnvar + 1e-5)**-0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    # Non-linearity
    h = torch.tanh(hpreact) # hidden layer
    # Linear layer 2
    logits = h @ W2 + b2 # output layer
    # cross entropy loss (same as F.cross_entropy(logits, Yb))
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes # subtract max for numerical stability
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdims=True)
    counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    # the code below is basically logprobs[0, Yb[0]] + logprobs[1, Yb[1]] + ... + logprobs[n-1, Yb[n-1]]
    loss = -logprobs[range(n), Yb].mean()

    # PyTorch backward pass
    for p in parameters:
        p.grad = None

    # tell pytorch to retain the grads of the intermediate tensors so we can compare them later
    for t in [logprobs, probs, counts, counts_sum, counts_sum_inv, # afaik there is no cleaner way
            norm_logits, logit_maxes, logits, h, hpreact, bnraw,
            bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
            embcat, emb]:
        t.retain_grad()
    loss.backward()
    
    # the derivative of the loss with respect to logprobs: shape of dlogprops should be same as logprobs (batch_size/n, vocab_size)
    # loss = -(a + b + c)/3. dloss/da = -1/3, dloss/db = -1/3, dloss/dc = -1/3: this is -1/n 
    dlogprobs = torch.zeros_like(logprobs)
    # set it to -1/n only for the values that were used in the calculation of the loss
    dlogprobs[range(n), Yb] = -1.0/n
    cmp('dlogprobs', dlogprobs, logprobs)

    # the derivative of the loss with respect to probs: shape of dprobs should be same as probs (batch_size, vocab_size)
    dprobs = dlogprobs * (1.0/probs) # dloss/dprobs = dloss/dlogprobs * dlogprobs/dprobs, derivative of log is 1/x (log is the natural log)
    cmp('dprobs', dprobs, probs)

    # dloss/dcounts_sum_inv = dloss/dprobs * dprobs/dcounts_sum_inv
    # shape of counts is (batch_size, vocab_size), shape of dprobs is  shape of dcounts_sum_inv should be (batch_size, 1)
    # the reason for the sum along the columns is that ???? some cheeky broadcasting and gradient calculation math
    dcounts_sum_inv = (dprobs * counts).sum(1, keepdims=True) # shape is (batch_size, 1)
    cmp('dcounts_sum_inv', dcounts_sum_inv, counts_sum_inv)

    dcounts = dprobs * counts_sum_inv # dloss/dcounts = dloss/dprobs * dprobs/dcounts, shape is (batch_size, vocab_size)
    dcounts_sum = -counts_sum**-2 * dcounts_sum_inv # dloss/dcounts_sum = dloss/dcounts_sum_inv * dcounts_sum_inv/dcounts_sum, shape is (batch_size/n, 1)
    cmp('dcounts_sum', dcounts_sum, counts_sum)

    # dloss/dcounts = dloss/dcounts_sum * dcounts_sum/dcounts + dloss/dcounts, shape is (batch_size, vocab_size)
    dcounts = torch.ones_like(counts) * dcounts_sum + dcounts 
    cmp('dcounts', dcounts, counts)

    # derivative of exp is exp, shape of dnorm_logits should be same as norm_logits (batch_size, vocab_size)
    dnorm_logits = norm_logits.exp() * dcounts
    cmp('dnorm_logits', dnorm_logits, norm_logits)

    dlogits = dnorm_logits.clone()
    dlogit_maxes = (-dnorm_logits).sum(1, keepdims=True)
    cmp('dlogit_maxes', dlogit_maxes, logit_maxes)

    dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
    cmp('dlogits', dlogits, logits)

    # derivative of loss with respect to h = dloss/dlogits * dlogits/dh
    # the backwards pass of a matrix multiplication is the matrix multiplication of the transpose of the matrix
    dh = dlogits @ W2.T # shape is (batch_size, n_hidden)
    dW2 = h.T @ dlogits # shape is (n_hidden, vocab_size)
    db2 = dlogits.sum(0) # shape is (vocab_size)
    cmp ('dh', dh, h)
    cmp ('dW2', dW2, W2)
    cmp ('db2', db2, b2)

    dhpreact = dh * (1.0 - h**2) # derivative of tanh is 1 - a^2, where a is the output of the tanh
    cmp('dhpreact', dhpreact, hpreact)

    dbngain = (bnraw * dhpreact).sum(0, keepdim=True) # shape is (1, n_hidden)
    dbnraw = bngain * dhpreact 
    dbnbias = dhpreact.sum(0, keepdim=True)
    cmp('dbngain', dbngain, bngain)
    cmp('dbnbias', dbnbias, bnbias)
    cmp('dbnraw', dbnraw, bnraw)

    dbndiff = bnvar_inv * dbnraw # shape is (batch_size, n_hidden)
    dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True) # shape is (1, n_hidden)
    cmp('dbnvar_inv', dbnvar_inv, bnvar_inv)
    cmp('dbndiff', dbndiff, bndiff)

    dbnvar = -0.5 * (bnvar + 1e-5)**-1.5 * dbnvar_inv
    cmp('dbnvar', dbnvar, bnvar)

    
