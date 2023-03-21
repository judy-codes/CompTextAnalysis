import random
import math
import torch
import torch.nn as nn
import numpy as np
from numpy import zeros, sign
from math import exp, log
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import argparse

torch.manual_seed(1701)

class SportsDataset(Dataset):
    def __init__(self, data):
        self.n_samples, self.n_features = data.shape
        # The first column is label, the rest are the features
        self.n_features -= 1 
        self.feature = torch.from_numpy(data[:, 1:].astype(np.float32)) # size [n_samples, n_features]
        self.label = torch.from_numpy(data[:, [0]].astype(np.float32)) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

def read_dataset(positive, negative, vocab):
    """
    Create a pytorch SportsDataset for the train and test data.

    :param positive: Positive examples file pointer
    :param negative: Negative examples file pointer
    :param vocab: Vocabulary words file pointer
    """

    # Below, we'll give you some code to get started because otherwise our test code will fail in ways that isn't very helpful.
    # 
    # Don't feel like you need to use this code if you have a better way of doing it (e.g., using sklearn), but you can use this if you want.
    # 
    # We'll need to allocate a matrix to store the data, but we can't start our matrix yet because we don't know the dimensions.
    # So first, what are the number of documents?  You'll need to change the below line, since you can't assume
    # that you have five documents in your dataset

    # Create a list of words in the vocabulary
    vocab1 = [line.split("\t")[0] for line in [line.strip() for line in vocab]]

    # Read data from files into memory
    positive_data = positive.readlines()
    negative_data = negative.readlines()

    num_docs = len(positive_data) + len(negative_data)

    # Create a matrix to store the data
    sportsdata = np.zeros((num_docs, len(vocab1)))

    counter = 0
    for label, data in [(1, positive_data), (0, negative_data)]:
        for line in data:
            l = line.strip().split()
            doc = defaultdict(int)
            for pair in [word.split(":") for word in l]:
                if len(pair) == 2:
                    word = pair[0]
                    count = pair[1]
                    if word in vocab1:
                        doc[word] += int(count)
            for word, count in doc.items():
                sportsdata[counter][vocab1.index(word)] += count
            sportsdata[counter][-1] = label
            counter += 1
    return sportsdata


class SimpleLogreg(nn.Module):
    def __init__(self, num_features):
        """
        Initialize the parameters you'll need for the model.

        :param num_features: The number of features in the linear model
        """
        super(SimpleLogreg, self).__init__()
        # Replace this with a real nn.Module
        self.linear = nn.Linear(num_features, 1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Compute the model prediction for an example.

        :param x: Example to evaluate
        """
        out = self.linear(x)
        out = self.Sigmoid(out)
        return out

    def evaluate(self, data):
        with torch.no_grad():
            y_predicted = self(data.feature)
            y_predicted_cls = y_predicted.round()
            acc = y_predicted_cls.eq(data.label).sum() / float(data.label.shape[0])
            return acc

#To answer question 2 on analysis
    # def get_top_parameters(self, k):
    #     """
    #     Returns the k largest and smallest parameters of the linear layer, sorted by absolute value.
    #     """
    #     weights = self.linear.weight.detach().numpy().flatten()
    #     abs_weights = np.abs(weights)
    #     sorted_indices = np.argsort(abs_weights)[::-1]
    #     top_indices = sorted_indices[:k]
    #     bottom_indices = sorted_indices[-k:]
    #     top_params = [(i, weights[i]) for i in top_indices]
    #     bottom_params = [(i, weights[i]) for i in bottom_indices]
    #     return top_params, bottom_params
        
def step(epoch, ex, model, optimizer, criterion, inputs, labels):
    """Take a single step of the optimizer, we factored it into a single
    function so we could write tests.  You should: A) get predictions B)
    compute the loss from that prediction C) backprop D) update the
    parameters

    There's additional code to print updates (for good software
    engineering practices, this should probably be logging, but printing
    is good enough for a homework).

    :param epoch: The current epoch
    :param ex: Which example / minibatch you're one
    :param model: The model you're optimizing
    :param inputs: The current set of inputs
    :param labels: The labels for those inputs
    """

    # Zero out gradients
    optimizer.zero_grad()
    
    # Get predictions
    outputs = model(inputs)
    
    # Compute loss from prediction
    loss = criterion(outputs, labels)
    
    # Backpropagation
    loss.backward()
    
    # Update parameters
    optimizer.step()

    # Analyze output
    if (ex+1) % 20 == 0:
        acc_train = model.evaluate(train)
        acc_test = model.evaluate(test)
        print(f'Epoch: {epoch+1}/{num_epochs}, Example {ex}, loss = {loss.item():.4f}, train_acc = {acc_train.item():.4f}, test_acc = {acc_test.item():.4f}')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    #''' Switch between the toy and REAL EXAMPLES
    argparser.add_argument("--positive", help="Positive class",
                           type=str, default="data/positive")
    argparser.add_argument("--negative", help="Negative class",
                           type=str, default="data/negative")
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="data/vocab")
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=5)
    argparser.add_argument("--batch", help="Number of items in each batch",
                           type=int, default=1)
    argparser.add_argument("--learnrate", help="Learning rate for SGD",
                           type=float, default=0.1)
    
    args = argparser.parse_args()
    
    sportsdata = read_dataset(open(args.positive), open(args.negative), open(args.vocab))
    # print(sportsdata)
    train_np, test_np = train_test_split(sportsdata, test_size=0.15, random_state=1234)
    train, test = SportsDataset(train_np), SportsDataset(test_np)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    logreg = SimpleLogreg(train.n_features)
    
    num_epochs = args.passes
    batch = args.batch
    total_samples = len(train)

    # Replace these with the correct loss and optimizer
    criterion = nn.MSELoss(reduction = 'mean')
    # criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(logreg.parameters(), lr=0.01)
    
    train_loader = DataLoader(dataset=train,
                              batch_size=batch,
                              shuffle=True,
                              num_workers=0)
    dataiter = iter(train_loader)

    # Iterations
    for epoch in range(num_epochs):
      for ex, (inputs, labels) in enumerate(train_loader):
        # Run your training process
        step(epoch, ex, logreg, optimizer, criterion, inputs, labels)

#To answer question 2 on analysis
    # vocab1 = [line.split("\t")[0] for line in [line.strip() for line in open(args.vocab)]]
    # top_params, bottom_params = logreg.get_top_parameters(k=10)
    # print("Top Parameters:")
    # for i, param in top_params:
    #     print(f"Weight index: {vocab1[i]}, value: {param}")
    # print("Bottom Parameters:")
    # for i, param in bottom_params:
    #     print(f"Weight index: {vocab1[i]}, value: {param}")
