# Judy Wang
from math import log, exp
from collections import defaultdict
import argparse

from numpy import mean

import nltk
nltk.download('treebank')
from nltk import FreqDist
from nltk.corpus import treebank
from nltk.util import bigrams
from nltk.tokenize import TreebankWordTokenizer

kLM_ORDER = 2
kUNK_CUTOFF = 3
kNEG_INF = -1e6

kSTART = "<s>"
kEND = "</s>"

def lg(x):
    return log(x) / log(2.0)

class BigramLanguageModel:

    def __init__(self, unk_cutoff, jm_lambda=0.6, add_k=0.1,
                 katz_cutoff=5, kn_discount=0.1, kn_concentration=1.0,
                 tokenize_function=TreebankWordTokenizer().tokenize,
                 normalize_function=lambda x: x.lower(),
                 dict={}, unigrams={}, bigrams={}):
        self._unk_cutoff = unk_cutoff
        self._jm_lambda = jm_lambda
        self._add_k = add_k
        self._katz_cutoff = katz_cutoff
        self._kn_concentration = kn_concentration
        self._kn_discount = kn_discount
        self._vocab_final = False

        self._tokenizer = tokenize_function
        self._normalizer = normalize_function
        
        # Add your code here!
        self._dict = dict
        self._unigrams = unigrams
        self._bigrams = bigrams

    def get_k(self):
        return self._add_k
    
    def get_jm_lambda(self):
        return self._jm_lambda

    def train_seen(self, word, count=1):
        """
        Tells the language model that a word has been seen @count times.  This
        will be used to build the final vocabulary.
        """
        assert not self._vocab_final, \
            "Trying to add new words to finalized vocab"
        
        # Add your code here! 
        if word in self._dict:
            self._dict[word] += count
        else:
            self._dict[word] = count

    def tokenize(self, sent):
        """
        Returns a generator over tokens in the sentence.  

        You don't need to modify this code.
        """
        for ii in self._tokenizer(sent):
            yield ii
        
    def vocab_lookup(self, word):
        """
        Given a word, provides a vocabulary representation.  Words under the
        cutoff threshold shold have the same value.  All words with counts
        greater than or equal to the cutoff should be unique and consistent.
        """
        assert self._vocab_final, \
            "Vocab must be finalized before looking up words"

        # Add your code here
        if word == kSTART or word == kEND:
            return word
        if word not in self._dict:
            return "UNK"
        elif self._dict[word] >= self._unk_cutoff:
            return word
        return "UNK"

    def finalize(self):
        """
        Fixes the vocabulary as static, prevents keeping additional vocab from
        being added
        """

        # You probably do not need to modify this code
        self._vocab_final = True

    def tokenize_and_censor(self, sentence):
        """
        Given a sentence, yields a sentence suitable for training or
        testing.  Prefix the sentence with <s>, replace words not in
        the vocabulary with <UNK>, and end the sentence with </s>.

        You should not modify this code.
        """
        yield self.vocab_lookup(kSTART)
        for ii in self._tokenizer(sentence):
            yield self.vocab_lookup(self._normalizer(ii))
        yield self.vocab_lookup(kEND)


    def normalize(self, word):
        """
        Normalize a word

        You should not modify this code.
        """
        return self._normalizer(word)


    def mle(self, context, word):
        """
        Return the log MLE estimate of a word given a context.  If the
        MLE would be negative infinity, use kNEG_INF
        """
        bigram = context, word
        if bigram in self._bigrams:
            if context not in self._unigrams:
                return kNEG_INF
            if self._unigrams[context] >= 0 and self._bigrams[bigram] >= 0:
                return lg(self._bigrams[bigram]/self._unigrams[context])
        return kNEG_INF

    def laplace(self, context, word):
        """
        Return the log MLE estimate of a word given a context.
        """
        bigram = context, word
        if bigram in self._bigrams:
            if context not in self._unigrams:
                return lg((self._bigrams[bigram]+1)/len(self._unigrams))
            if self._unigrams[context] >= 0 and self._bigrams[bigram] >= 0:
                return lg((self._bigrams[bigram]+1)/(self._unigrams[context]+len(self._unigrams)))
        elif context in self._unigrams:
            return lg(1/(self._unigrams[context]+len(self._unigrams)))
        return lg(1.0/len(self._unigrams))

    def jelinek_mercer(self, context, word):
        """
        Return the Jelinek-Mercer log probability estimate of a word
        given a context; interpolates context probability with the
        overall corpus probability.
        """
        bigram = context, word
        first = 0
        sec = 0
        jm = self.get_jm_lambda()
        if bigram in self._bigrams:
            if context not in self._unigrams:
                first = 0
            if self._unigrams[context] >= 0 and self._bigrams[bigram] >= 0:
                first = self._bigrams[bigram]/self._unigrams[context]
        if word in self._unigrams:
            total = 0
            for w in self._unigrams:
                if(w != kSTART): 
                    total += self._unigrams[w]
            sec = self._unigrams[word]/total
        what = (first*jm) + ((1-jm)*(sec))
        return lg(what)

    def kneser_ney(self, context, word):
        """
        Return the log probability of a word given a context given
        Kneser Ney backoff
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        return 0.0

    def add_k(self, context, word):
        """
        Additive smoothing, i.e. add_k smoothing, assuming a fixed k
        hyperparameter.
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        bigram = context, word
        k = self.get_k()
        if bigram in self._bigrams:
            if context not in self._unigrams:
                return lg((self._bigrams[bigram]+k)/(len(self._unigrams)*k))
            if self._unigrams[context] >= 0 and self._bigrams[bigram] >= 0:
                return lg((self._bigrams[bigram]+k)/(self._unigrams[context]+(len(self._unigrams)*k)))
        elif context in self._unigrams:
            return lg(k/(self._unigrams[context]+(len(self._unigrams)*k)))
        return lg(k/(len(self._unigrams)*k))

    def add_train(self, sentence):
        """
        Add the counts associated with a sentence.
        """
        for context, word in bigrams(self.tokenize_and_censor(sentence)):
            bigram = context, word
            if bigram not in self._bigrams:
                self._bigrams[bigram] = 1
            else:
                self._bigrams[bigram] += 1

        for word in self.tokenize_and_censor(sentence):
            if word not in self._unigrams:
                self._unigrams[word] = 1
            else: 
                self._unigrams[word] += 1
            
    def perplexity(self, sentence, method):
        """
        Compute the perplexity of a sentence given a estimation method

        You do not need to modify this code.
        """
        return 2.0 ** (-1.0 * mean([method(context, word) for context, word in \
                                    bigrams(self.tokenize_and_censor(sentence))]))

    def sample(self, method, samples=25):
        """
        Sample words from the language model.
        
        @arg samples The number of samples to return.
        """
        # Modify this code to get extra credit.  This should be
        # written as an iterator.  I.e. yield @samples times followed
        # by a final return, as in the sample code.

        for ii in xrange(samples):
            yield ""
        return

# You do not need to modify the below code, but you may want to during
# your "exploration" of high / low probability sentences.
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--jm_lambda", help="Parameter that controls " + \
                           "interpolation between unigram and bigram",
                           type=float, default=0.6, required=False)
    argparser.add_argument("--add_k", help="Add k value " + \
                           "for pseudocounts",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--unk_cutoff", help="How many times must a word " + \
                           "be seen before it enters the vocabulary",
                           type=int, default=2, required=False)    
    argparser.add_argument("--katz_cutoff", help="Cutoff when to use Katz " + \
                           "backoff",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--lm_type", help="Which smoothing technique to use",
                           type=str, default='mle', required=False)
    argparser.add_argument("--brown_limit", help="How many sentences to add " + \
                           "from Brown",
                           type=int, default=-1, required=False)
    argparser.add_argument("--kn_discount", help="Kneser-Ney discount parameter",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--kn_concentration", help="Kneser-Ney concentration parameter",
                           type=float, default=1.0, required=False)
    argparser.add_argument("--method", help="Which LM method we use",
                           type=str, default='laplace', required=False)
    
    args = argparser.parse_args()    
    lm = BigramLanguageModel(kUNK_CUTOFF, jm_lambda=args.jm_lambda,
                             add_k=args.add_k,
                             katz_cutoff=args.katz_cutoff,
                             kn_concentration=args.kn_concentration,
                             kn_discount=args.kn_discount)

    for ii in nltk.corpus.brown.sents():
        for jj in lm.tokenize(" ".join(ii)):
            lm.train_seen(lm._normalizer(jj))

    print("Done looking at all the words, finalizing vocabulary")
    lm.finalize()

    sentence_count = 0
    for ii in nltk.corpus.brown.sents():
        sentence_count += 1
        lm.add_train(" ".join(ii))

        if args.brown_limit > 0 and sentence_count >= args.brown_limit:
            break

    print("Trained language model with %i sentences from Brown corpus." % sentence_count)
    assert args.method in ['kneser_ney', 'mle', 'add_k', \
                           'jelinek_mercer', 'good_turing', 'laplace'], \
      "Invalid estimation method"

    sent = input()
    while sent:
        print("#".join(str(x) for x in lm.tokenize_and_censor(sent)))
        print(lm.perplexity(sent, getattr(lm, args.method)))
        sent = input()

    # for sent in nltk.corpus.treebank.sents():
    #     print("#".join(str(x) for x in lm.tokenize_and_censor(str(sent))))
    #     print(lm.perplexity(str(sent), getattr(lm, args.method)))