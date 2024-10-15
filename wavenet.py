## lot of boilerplate code is the same as the mlp_structured.py file

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

#================================================================================================

def build_dataset(words):
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
block_size = 8 # number of characters do we need to predict the next one
n1 = int(len(words) * 0.8)
n2 = int(len(words) * 0.9)
Xtr, Ytr = build_dataset(words[:n1])
Xdv, Ydv = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647) # global generator for reproducibility

#================================================================================================

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
    
#================================================================================================

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
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            x_mean = x.mean(dim=dim, keepdim=True)    
            x_var = x.var(dim=dim, keepdim=True)
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

#================================================================================================
    
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

#================================================================================================
    
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn(num_embeddings, embedding_dim)
    
    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        return [self.weight]

#================================================================================================

class FlattenConsecutive:
    
    # only flatten n consecutive elements
    def __init__(self, n):
        self.n = n
    
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C * self.n)

        # if the middle dimension is 1, just return a 2D tensor
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []

#================================================================================================
# Organize the layers in a sequential manner

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        # return a list of all the parameters of the layers
        return [p for layer in self.layers for p in layer.parameters()]

#================================================================================================
# Define the model

n_embd = 24 # the dimensionality of the character embedding vectors
n_hidden = 128 # number of neurons in the hidden layer of the MLP
vocab_size = len(stoi)

'''
    In the previous, we flattend all the characters in a block into a 2D tensor.
    But now we do this - (1 2) (3 4) (5 6) (7 8), only flatten 2 consecutive characters
'''

model = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size, bias=False)
])

# with torch.no_grad():
#     # make the last layer less confident
#     layers[-1].gamma *= 0.1
#     for layer in layers[:-1]:
#         if isinstance(layer, Linear):
#             # counter the squashing effects of the tanh functions, 5/3 is just a good number for some reason
#             layer.weight *= 1.0

parameters = model.parameters()
for p in parameters:
    p.requires_grad = True

#================================================================================================
# Train the model

max_steps = 200000
lossi = []
batch_size = 32
for i in range(max_steps):
    # the minibatch size is 32
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # minibatch

    # forward pass
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    # backward pass
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


#================================================================================================
# Evaluate the model

def evaluate_loss(X, Y):
    logits = model(X)
    loss = F.cross_entropy(logits, Y)
    return loss.item()

# Example usage:
## TODO: Set the model to evaluation mode properly
for layer in model.layers:
    layer.training = False
train_loss = evaluate_loss(Xtr, Ytr)
validation_loss = evaluate_loss(Xdv, Ydv)

print(f'Training loss: {train_loss}')
print(f'Validation loss: {validation_loss}')

#================================================================================================

# sample from the model
def sample(length):
    for _ in range(length):
        out = []
        context = [0]*block_size
        while True:
            logits = model(torch.tensor([context]))
            # Get the probabilities and sample from them
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