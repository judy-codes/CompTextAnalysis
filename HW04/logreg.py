import random
import numpy as np
from math import exp, log
from collections import defaultdict

import argparse

kSEED = 1701
kBIAS = "BIAS_CONSTANT"

random.seed(kSEED)


def sigmoid(score, threshold=20.0):
    """
    Prevent overflow of exp by capping activation at 20.
    :param score: A real valued number to convert into a number between 0 and 1
    """

    if abs(score) > threshold:
        score = threshold * np.sign(score)

    activation = exp(score)
    return activation / (1.0 + activation)


class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, label, words, vocab, df):
        """
        Create a new example
        :param label: The label (0 / 1) of the example
        :param words: The words in a list of "word:count" format
        :param vocab: The vocabulary to use as features (list)
        """
        self.nonzero = {vocab.index(kBIAS): 1}
        self.y = label  
        self.x = np.zeros(len(vocab))
        for word, count in [x.split(":") for x in words]:
            if word in vocab:
                assert word != kBIAS, "Bias can't actually appear in document"
                self.x[vocab.index(word)] += float(count)
                self.nonzero[vocab.index(word)] = word
        self.x[0] = 1


class LogReg:
    def __init__(self, num_features, mu, step):
        """
        Create a logistic regression classifier
        :param num_features: The number of features (including bias)
        :param mu: Regularization parameter (for extra credit)
        :param step: A function that takes the iteration as an argument (the default is a constant value)
        """

        self.dimension = num_features
        self.beta = np.zeros(num_features)
        self.mu = mu
        self.step = step
        self.last_update = np.zeros(num_features)

        assert self.mu >= 0, "Regularization parameter must be non-negative"

    def progress(self, examples):
        """
        Given a set of examples, compute the probability and accuracy
        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """

        logprob = 0.0
        num_right = 0
        for ii in examples:
            p = sigmoid(self.beta.dot(ii.x))
            if ii.y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            if self.mu > 0:
                logprob -= self.mu * np.sum(self.beta ** 2)

            # Get accuracy
            if abs(ii.y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples))

    def ranks(self, examples, imp, best):
        """
        Given a set of examples, compute the probability and accuracy
        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """
        best_hoc = {}
        best_bas = {}
        if imp == 1:
            for ii in examples:
                p = sigmoid(self.beta.dot(ii.x))
                if abs(ii.y - p) < 0.5:
                    max = np.argmax(ii.x)
                    word = ii.nonzero[max]
                    if ii.y == 1:
                        best_bas[word] = abs(ii.y - p)
                    else:
                        best_hoc[word] = abs(ii.y - p)

        #Adds all of the word's probabilities and those who have the least amount are the most correlated
        elif imp == 2:
            for ii in examples:
                p = sigmoid(self.beta.dot(ii.x))
                if abs(ii.y - p) < 0.5:
                    for word in ii.nonzero.values():
                        if ii.y == 1 and word not in best_bas:
                            best_bas[str(word)] = abs(ii.y - p)
                        elif ii.y == 1:
                            best_bas[str(word)] += abs(ii.y - p)
                        elif word not in best_hoc:
                            best_hoc[str(word)] = abs(ii.y - p)
                        else:
                            best_hoc[str(word)] += abs(ii.y - p)
        else: 
            return
        # Convert dictionary to list of key-value pairs
        hoc = list(best_hoc.items())
        bas = list(best_bas.items())

        # Sort list based on values
        hoc.sort(key=lambda x: x[1])
        bas.sort(key=lambda x: x[1])
        
        if best: 
            # Get the first 10 items (i.e. the 10 lowest values)
            bw_hoc = [t[0] for t in hoc[:10]]
            bw_bas = [t[0] for t in bas[:10]]
            return bw_hoc, bw_bas
        else:
            ww_hoc = [t[0] for t in hoc[-10:]]
            ww_bas = [t[0] for t in bas[-10:]]
            return ww_hoc, ww_bas
        

        
    
    def sg_update(self, train_example, iteration,
                  lazy=False, use_tfidf=False):
        """
        Compute a stochastic gradient update to improve the log likelihood.
        :param train_example: The example to take the gradient with respect to
        :param iteration: The current iteration (an integer)
        :param use_tfidf: A boolean to switch between the raw data and the tfidf representation
        :return: Return the new value of the regression coefficients
        """
        features = train_example.x
        curr_pred = train_example.y

        strength = self.beta.dot(features)
        new_pred = sigmoid(strength)
        gradient = (curr_pred - new_pred) * features
        
        learning_rate = self.step(iteration)
        self.beta += learning_rate * gradient

        return self.beta

    def finalize_lazy(self, iteration):
        """
        After going through all normal updates, apply regularization to
        all variables that need it.
        Only implement this function if you do the extra credit.
        """

        return self.beta

def read_dataset(positive, negative, vocab, test_proportion=.1):
    """
    Reads in a text dataset with a given vocabulary
    :param positive: Positive examples
    :param negative: Negative examples
    :param vocab: A list of vocabulary words
    :param test_proprotion: How much of the data should be reserved for test
    """
    df = [float(x.split("\t")[1]) for x in open(vocab, 'r') if '\t' in x]
    vocab = [x.split("\t")[0] for x in open(vocab, 'r') if '\t' in x]
    assert vocab[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vocab[0]

    train = []
    test = []
    for label, input in [(1, positive), (0, negative)]:
        for line in open(input):
            ex = Example(label, line.split(), vocab, df)
            if random.random() <= test_proportion:
                test.append(ex)
            else:
                train.append(ex)

    # Shuffle the data so that we don't have order effects
    random.shuffle(train)
    random.shuffle(test)

    return train, test, vocab

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mu", help="Weight of L2 regression",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--step", help="Initial SG step size",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--positive", help="Positive class",
                           type=str, default="data/data/positive", required=False)
    argparser.add_argument("--negative", help="Negative class",
                           type=str, default="data/data/negative", required=False)
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="data/data/vocab", required=False)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)
    argparser.add_argument("--ec", help="Extra credit option (df, lazy, or rate)",
                           type=str, default="")

    args = argparser.parse_args()
    train, test, vocab = read_dataset(args.positive, args.negative, args.vocab)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    if args.ec != "rate":
        lr = LogReg(len(vocab), args.mu, lambda x: args.step)
    else:
        # Modify this code if you do learning rate extra credit
        raise NotImplementedError

    # Iterations
    update_number = 0
    for pp in range(args.passes):
        for ii in train:
            update_number += 1
            # Do we use extra credit option
            if args.ec == "df":
                lr.sg_update(ii, update_number, use_tfidf=True)
            elif args.ec == "lazy":
                lr.sg_update(ii, update_number, lazy=True)
            else:
                lr.sg_update(ii, update_number)

            if update_number % 5 == 1:
                train_lp, train_acc = lr.progress(train)
                ho_lp, ho_acc = lr.progress(test)
                print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
                      (update_number, train_lp, ho_lp, train_acc, ho_acc))
                
    # Final update with empty example
    lr.finalize_lazy(update_number)
    print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
          (update_number, train_lp, ho_lp, train_acc, ho_acc))

#For question number 3 and 4 in README 
    # hoc, bas = lr.ranks(train, 2, False)
    # print(hoc)
    # print(bas)

#For question number 2 in README
    # all_acc = []
    # last_acc = 0;
    # done = False
    # update_number = 0

    # while True:
    #     for ii in train:
    #         update_number += 1
    #         # Do we use extra credit option
    #         if args.ec == "df":
    #             lr.sg_update(ii, update_number, use_tfidf=True)
    #         elif args.ec == "lazy":
    #             lr.sg_update(ii, update_number, lazy=True)
    #         else:
    #             lr.sg_update(ii, update_number)

    #         train_lp, train_acc = lr.progress(train)
    #         ho_lp, ho_acc = lr.progress(test)
    #         print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
    #                 (update_number, train_lp, ho_lp, train_acc, ho_acc))
            
    #         if ho_acc - last_acc <= 0:
    #             # Final update with empty example
    #             lr.finalize_lazy(update_number)
    #             print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
    #                 (update_number, train_lp, ho_lp, train_acc, ho_acc))
    #             done = True
    #         last_acc = ho_acc
    #     if done:
    #         break