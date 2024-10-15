'''
Here we predict just one character based on the previous character. This is a simple example to start learning. 
'''

import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Ensure the file path is correct
file_path = os.path.join(os.path.dirname(__file__), 'names.txt')
words = open(file_path, 'r').read().splitlines()

# this is a tensor of 28 x 28, each element is a int and represents the count of the bigram
# we use 28 because we have 26 letters + start and end token
N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
# string to integer mapping
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0

# function to invert the mapping from integer to string
itos = {i:s for s, i in stoi.items()}
for w in words:
    # add start and end token to each word, so that the model knows when to start and stop predicting
    # each word has two pieces of information:
        # 1. it encodes what characters should come next when generating a name
        # 2. it also encodes when the name should end

    '''
    Example:
    w = 'john'
    chs = ['<S>', 'j', 'o', 'h', 'n', '<E>']
    bigram = ('<S>', 'j')
    bigram = ('j', 'o')
    bigram = ('o', 'h')
    bigram = ('h', 'n')
    bigram = ('n', '<E>')

    This allows us to count how many times a character is a starting or ending character in a name
    '''
    chs = ['.'] + list(w) + ['.']
    # iterate through each word with a sliding window of size 2
    # zip(w, w[1:]) will create a tuple of two characters (cool python trick)
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        # increment the count of the bigram
        N[ix1, ix2] += 1

# the first row of the matrix N is the count of the start token (how many times a character is the start of a name)
# the first column of the matrix N is the count of the end token (how many times a character is the end of a name)
# using the multinomial function, we can sample from the distribution of the start token to sample the first character of a name
# then we can go to row corresponding to the sampled character and sample the next character based on the distribution of the first character
# we can continue this process until we sample the end token

# first prepare the matrix of probabilities for each character
# add 1 to the count to avoid division by zero and 0 probabilities
P = (N + 1).float()
'''
get the sum of each row (dim = 1) and divide each element of the row by the sum
P is 27 x 27, P.sum(dim=1, keepdim=True) is 27 x 1 (column vector)
The broading casting rules of numpy and pytorch allow us to divide a 27 x 27 matrix by a 27 x 1 column vector
because the column vector is broadcasted to a 27 x 27 matrix (each column is the same)
then we can do an element-wise division (just like normal element wise addition)
in this way, every element in each row of P is divided by the sum of the row
'''

'''
importance of the keepdim parameter

If we don't use keepdim=True, the sum will be a 1D tensor of size 27
P is a 27 x 27 matrix, so the division will not work
27 x 27
1 x 27 (internally, pytorch turns just 27 into 1 x 27), then broadcasts to 27 x 27, where each row is the same
Now if we do element wise division, the first element of the first row will be divided by the sum of the first row
But the second element of the first row will be divided by the sum of the second row, messing up the probabilities
'''
P /= P.sum(dim=1, keepdim=True)

ix = 0 # start from first character
# we use a fixed seed to ensure that the results are reproducible
g = torch.Generator().manual_seed(2147483647)
for i in range(20):
    out = []
    ix = 0
    while True:
        p = P[ix]
        # uniform distribution, untrained model, gives a bit more garbage than the trained model
        #p = torch.ones(27) / 27.0
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

    #print(''.join(out))

'''
Now we have a simple model that can generate names based on the bigram model.
We determine the probability of the next character based on the previous character, and use a multinomial distribution to sample the next character.
We also determined a nice way to figure out when to stop generating characters by using the start and end tokens.

Now we need to train the model to generate better names.
First we need to figure out how to evaluate the model. (how good is it)

To evaluate the model, we can calculated the likelihood of the data given the model. 
So the likelihood that the model generates the data that we have. (would need to multiply the probabilities of each bigram)
This is a very small number, so we can use the log likelihood instead (log works because it is monotonic).
using log rules (log(a*b) = log(a) + log(b)), we can convert the multiplication of probabilities to addition of log probabilities

This log likelihood / (num of bigrams) will be our loss function, and it is a negative number (we want it to bring it towards 0 since log(1) = 0).
Generally, we want to minimize the loss function instead of maximizing, so we can just negate the log likelihood to get the loss function.
'''


log_likelihood = 0
n = 0

for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        log_prob = torch.log(prob)
        log_likelihood += log_prob
        n += 1
print(f'log likelihood: {log_likelihood / n}')

# create the training set of bigrams (x, y)
# the model will predict y based on x
# x is the previous character, y is the next character
# xs is the previous character, ys is the next character from the training set
# these will be used to train the neural network
xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)


'''
    We cannot feed the character indices directly to the neural network.
    We need to convert them to one-hot vectors of floats.
    One-hot vectors are vectors with all zeros except for one element which is 1.
    e.g. to encode the integer 3 in a one-hot vector of size 5, we would have [0, 0, 0, 1, 0]
'''
# initialize the weights and biases of the neural network that is single layer and has 27 input and output units
g = torch.Generator().manual_seed(2147483647)
# requires_grad=True tells pytorch to keep track of the gradients of the weights (cool beans)
W = torch.randn((27, 27), generator=g, requires_grad=True)

'''
    For each input character, we want to output the 27 probabilities of the next character.
    We can do this by multiplying the one-hot vector of the input character with the weights matrix (this results in log-counts)
    We can then exponentiate the results and divide each element by the sum of the row to get the probabilities.

    ys stores the correct next character for each input
    we can find what the model predicted by:
        e.g. xs = [0, 5, 13], ys = [5, 13, 0]
        probs[0, 5] is the probability of the model predicting 5 given 0
        probs[1, 13] is the probability of the model predicting 13 given 5

    We know how calculate the loss using the negative log likelihood (above)
    We can average the loss over all the bigrams to determine the loss function for this pass
    Then we can use backpropagation to update the weights of the neural network
'''

for k in range(100):

    # @ is the matrix multiplication operator in pytorch
    # below is the forward pass of the neural network
    xenc = F.one_hot(xs, num_classes=27).float() # input to neural network
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)
    loss = -probs[torch.arange(len(ys)), ys].log().mean()
    print(loss)

    '''
        Pytorch keeps track of the gradients of the weights (W) when we do the forward pass.
        It keeps track of all the operations that were done to calculate the loss and uses the chain rule to calculate the gradients of the weights.
        It does what we did with micrograd, but it does it automatically for us. (cool beans)
        We can then use the gradients to update the weights of the neural network.
    '''
    # backward pass
    # set it to None to avoid pytorch from accumulating the gradients
    W.grad = None
    loss.backward() # calculate the gradients of the weights and updates W.grad
    # how did we choose 0.1 as the learning rate? The smaller the learning rate, the less likely we are to overshoot the minimum (but it will take longer to converge)
    W.data -= 50 * W.grad # update the weights using the gradients

'''
The loss we are looking for is what we calculated as the loss of the bigram model where we used counts to determine probabilities.
The gradient based approach however is much more flexible and can be used for more complex models.
'''

# sample from the gradient trained model

for i in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum()
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))