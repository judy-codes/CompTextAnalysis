# Author: Judy Wang :)
# Date: 1/20/2023

# Use word_tokenize to split raw text into words
from string import punctuation
import nltk
import re
from nltk.tokenize import word_tokenize


class LimerickDetector:

    def __init__(self):
        """
        Initializes the object to have a pronunciation dictionary available
        """
        self._pronunciations = nltk.corpus.cmudict.dict()

    def num_syllables(self, word):
        """
        Returns the number of syllables in a word.  If there's more than one
        pronunciation, take the shorter one.  If there is no entry in the
        dictionary, return 1.
        """
        if word not in self._pronunciations:
            return 1
        else: 
            pron = self._pronunciations[word]
            min_syllables = 100000000
            for p in pron:
                num_syllables = sum(part[-1].isdigit() for part in p)
                if num_syllables < min_syllables:
                    min_syllables = num_syllables
            return min_syllables

    def rhyming_part(self, word):
        """
        Returns the set of pronouncations that a count as valid sounds for a rhyme. This means that it cut offs
        the first constant sound(s) for the word
        """
        pron = self._pronunciations[word]
        all_rhy = []
        for p in pron:
            for i in range(len(p)):
                if p[i][-1].isdigit():
                    all_rhy.append(p[i:])
                    break
        return all_rhy

    def rhymes(self, a, b):
        """
        Returns True if two words (represented as lower-case strings) rhyme,
        False otherwise.
        """
        rhy_a = self.rhyming_part(a)
        rhy_b = self.rhyming_part(b)

        shorter = rhy_a if len(rhy_a[0]) < len(rhy_b[0]) else rhy_b
        longer = rhy_a if shorter == rhy_b else rhy_b

        for s in shorter:
            for l in longer:
                if l[-len(s):] == s:
                    return True
        return False

    def is_limerick(self, text):
        """
        Takes text where lines are separated by newline characters.  Returns
        True if the text is a limerick, False otherwise.

        A limerick is defined as a poem with the form AABBA, where the A lines
        rhyme with each other, the B lines rhyme with each other (and not the A
        lines).

        (English professors may disagree with this definition, but that's what
        we're using here.)
        """
        text = re.sub(r'[^\w\s]','', text)
        split = list(filter(str.strip, text.split('\n')))
        if len(split) < 5:
            return False
        for i in range(0,len(split)-len(split)%5-1,4):
            if self.rhymes(word_tokenize(split[i])[-1], word_tokenize(split[i+1])[-1]) == False or self.rhymes(word_tokenize(split[i+2])[-1], word_tokenize(split[i+3])[-1]) == False or self.rhymes(word_tokenize(split[i])[-1], word_tokenize(split[i+4])[-1]) == False:
                return False
        return True

if __name__ == "__main__":
    buffer = ""
    inline = " "
    while inline != "":
        buffer += "%s\n" % inline
        inline = input()

    ld = LimerickDetector()
    print("%s\n-----------\n%s" % (buffer.strip(), ld.is_limerick(buffer)))