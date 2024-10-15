## this code is just the same code from mlp.py but with more structre so we can actually use it

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

g = torch.Generator().manual_seed(2147483647)

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn(fan_in, fan_out, generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias  
        
        return self.out

    def parameters(self):
        return [self.weight] if self.bias is None else [self.weight, self.bias]
    

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        #parameters - trained with backpropagation
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        #running statistics - not trained with backpropagation 
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    # the call method is called when the object is called like a function
    def __call__(self, x):
        # first calculate the mean and variance of the batch
        if self.training:
            # batch mean and variance
            x_mean = x.mean(dim=0, keepdim=True)    
            x_var = x.var(dim=0, keepdim=True)
        else:
            x_mean = self.running_mean
            x_var = self.running_var

        # normalize the batch
        x_hat = (x - x_mean) / torch.sqrt(x_var + self.eps)
        self.out = self.gamma * x_hat + self.beta

        # update the running stats
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
    
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []
    

n_embd = 10
n_hidden = 200
block_size = 3
vocab_size = len(stoi)

C = torch.randn(vocab_size, n_embd, generator=g)

layers = [
    Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size, bias=False), BatchNorm1d(vocab_size), 
]

with torch.no_grad():
    # make the last layer less confident
    layers[-1].gamma *= 0.1
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            # counter the squashing effects of the tanh functions, 5/3 is just a good number for some reason
            layer.weight *= 1.0

parameters = [C] + [p for layer in layers for p in layer.parameters()]
for p in parameters:
    p.requires_grad = True

max_steps = 2000
lossi = []
batch_size = 32
for i in range(max_steps):
    # the minibatch size is 32
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # minibatch

    # forward pass
    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        # the output of the previous layer is the input to the next layer
        x = layer(x)
    loss = F.cross_entropy(x, Yb)

    # backward pass
    # we retain the gradients of the output of the layers because we need them for the backward pass
    for layer in layers:
        layer.out.retain_grad()
    
    for p in parameters:
        p.grad = None
    loss.backward()

    # update the parameters
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
    #print(loss.item())
    lossi.append(loss.log10().item())
    # if i > 1000:
    #     break

def evaluate_loss(X, Y):
    emb = C[X]
    x = emb.view(emb.shape[0], -1)
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Y)
    return loss.item()

# Example usage:
for layer in layers:
    layer.training = False
train_loss = evaluate_loss(Xtr, Ytr)
validation_loss = evaluate_loss(Xdv, Ydv)

print(f'Training loss: {train_loss}')
print(f'Validation loss: {validation_loss}')

# sample from the model
def sample(length):
    for _ in range(length):
        out = []
        context = [0]*block_size
        while True:
            emb = C[torch.tensor([context])]
            x = emb.view(emb.shape[0], -1)
            for layer in layers:
                x = layer(x)
        
            # Get the probabilities and sample from them
            logits = x
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()

            # Update the context
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        print(''.join(itos[i] for i in out))

# Example usage:
sample(length=20)  # Generates a name of up to 20 character