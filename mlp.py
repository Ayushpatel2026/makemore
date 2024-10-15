import torch
import torch.nn.functional as F
import os
import random

import matplotlib.pyplot as plt

file_path = os.path.join(os.path.dirname(__file__), 'names.txt')
words = open(file_path, 'r').read().splitlines()


# do the mapping from character to integer
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

#training split, 80% of the data, used to train the model using backpropagation
# dev/validation split, 10% of the data, used to tune hyperparameters
# test split, 10% of the data
'''
 X will be a list of context vectors, each of size block_size
 Y will be a list of target characters
 e.g. for word emma and block_size = 3
 X = [[0, 0, 0], [0, 0, 5], [0, 5, 13], [5, 13, 13], [13, 13, 1]]
 Y = [5, 13, 13, 1, 0]
'''
def build_dataset(words):
    block_size = 3 # number of characters do we need to predict the next one
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(len(words) * 0.8)
n2 = int(len(words) * 0.9)
Xtr, Ytr = build_dataset(words[:n1])
Xdv, Ydv = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

'''
Let us make a neural network that takes in a context vector and predicts the next character
C is a matrix of embeddings for each character
emb is a tensor of embeddings for each context vector
W1 is the weight matrix for the first layer
b1 is the bias vector for the first layer
h is the output of the first layer
emb has dimensions (N, block_size, 2)
W1 has dimensions (2 * block_size, 100)
Emb must be a (N, 2 * block_size) tensor for the matrix multiplication to work, we use .view for this
The result of the matrix multiplication is a (N, 100) tensor and this is added to the bias vector b1

    N 100 using the broadcasting rules indicate that the element wise addition here is what we want
    1 100 means 

Tanh squashes the output of the first layer to be between -1 and 1
'''
n_embd = 10
block_size = 3
n_hidden = 200
vocab_size = len(stoi)
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator=g, requires_grad=True)
# this layer will have 100 neurons and 6 inputs
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g, requires_grad=True)
W1.data = W1.data * (5/3)/(n_embd * block_size)**0.5 # make the initialization scale-invariant
b1 = torch.randn(n_hidden, generator=g, requires_grad=True)
b1.data = b1.data * 0.1
# second layer will have 27 neurons and 100 inputs (output of the first layer)
W2 = torch.randn((n_hidden, vocab_size), generator=g, requires_grad=True)
W2.data *= 0.1
b2 = torch.randn(vocab_size, generator=g, requires_grad=True)
b2.data *= 0.1

'''
    Batch normalization is a technique that normalizes the input to a layer so that the mean is 0 and the standard deviation is 1.
    But we still want to learn the mean and standard deviation for the layer
    So we have two learnable parameters for each layer, the gain and the bias.
    The gain is multiplied by the normalized input and the bias is added to it.
    The mean and standard deviation are calculated for each minibatch and are used to normalize the input.
    We also keep a running average of the mean and standard deviation for the entire dataset.
    This is used to normalize the input during evaluation.
'''

batch_normal_gain = torch.randn((1, n_hidden), requires_grad=True) * 0.1 + 1.0
batch_normal_bias = torch.randn((1, n_hidden), requires_grad=True) * 0.1
bnmean_running = torch.zeros(1, n_hidden, requires_grad=True)
bnstd_running = torch.ones(1, n_hidden, requires_grad=True)

parameters = [C, W1, W2, b1, b2, batch_normal_gain, batch_normal_bias]

# potential learning rates
learning_rate_exp = torch.linspace(-3, 0, 1000)
learning_rates = 10 ** learning_rate_exp
lri = []
lossi = []
batch_size = 32
for i in range(200000):
    # the minibatch size is 32
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))

    emb = C[Xtr[ix]]
    hpreact = emb.view(emb.shape[0], n_embd * block_size) @ W1 #+ b1
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)
    hpreact = batch_normal_gain * (hpreact - bnmeani) / bnstdi + batch_normal_bias

    with torch.no_grad():
        # the 0.001 here is proportional to batch size, if batch size is bigger, we can have a bigger value here because each batch is more representative of the entire dataset
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

    h = torch.tanh(hpreact)

    logits = h @ W2 + b2
    # counts = logits.exp()
    # probs = counts / counts.sum(dim=1, keepdim=True)
    # loss = -probs[torch.arange(len(Y)), Y].log().mean()
    # this line does what the 3 lines above do, but much more efficiently
    loss = F.cross_entropy(logits, Ytr[ix])
    #print(loss.item())
    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    # learning rate will increase as the number of iterations increase
    # lrs = learning_rates[i]
    lrs = 0.1
    if i > 100000:
        lrs = 0.01
    for p in parameters:
        p.data += -lrs * p.grad
    
    # lri.append(lrs)
    # lossi.append(loss.item())

emb = C[Xtr]
hpreact = emb.view(emb.shape[0], n_embd * block_size) @ W1 #+ b1
hpreact = batch_normal_gain * (hpreact - bnmean_running) / bnstd_running + batch_normal_bias
h = torch.tanh(hpreact)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
print("Total loss on training", loss.item())
    

#loss after training
emb = C[Xdv]
hpreact = emb.view(emb.shape[0], n_embd * block_size) @ W1 #+ b1
hpreact = batch_normal_gain * (hpreact - bnmean_running) / bnstd_running + batch_normal_bias
h = torch.tanh(hpreact)
h = torch.tanh(emb.view(emb.shape[0], n_embd * block_size) @ W1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydv)
print("Total loss on dev", loss.item())

#sample from the model
for _ in range(20):
    out = []
    context = [0]*block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))

'''
Modifications to the og model that were made to improve performance:
When the weights and biases are initialized, if they are making confident mispredictions, the loss will be very high at the start
This results in a hockey stick shaped loss curve. We can avoid this by scaling the weights and biases and squashing them. This allows us to do more actual learning.

 We also want the activations to be roughly normally distributed. We can do this by normalizing the activations to have a mean of 0 and a standard deviation of 1.
 The batch normalization layer does this for us. We can also use the running average of the mean and standard deviation to normalize the activations during evaluation.
 This only really works for small networks, not for deeper networks.

 Also, pytorch as built in linear layers and batch normalization layers that we can use instead of manually defining the weights and biases.
 The pytorch documentation has more info on this. 
'''

# this function checks our manual gradients against what pytorch calculates
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    max_diff = (dt - t.grad).abs().max().item()

    print(f'{s:15s} | exact: {str(ex):5s} | approx: {str(app):5s} | max_diff: {max_diff}')