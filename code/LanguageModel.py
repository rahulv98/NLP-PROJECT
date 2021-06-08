from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from util import *

from nltk import ngrams
from nltk.util import pad_sequence
import numpy as np
import itertools
from collections import Counter
from scipy import stats


class NgramModel():
    def __init__(self, n, m=4, k=0, smooth=None):
        if n > 3 or n < 1:
            raise Exception("Only n = 1, 2, 3 are accepted")
        
        self.n = n
        self.k = k
        self.m = m
        self.index = [{}, {}, {}]
        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.smooth = smooth

    def segmentSentences(self, text):
        """
        Call the required sentence segmenter
        """
        return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        """
        Call the required tokenizer
        """
        return self.tokenizer.pennTreeBank(text)

    def buildIndex(self, docs):
        """
        Build counts index and fit log curve if smoooth method is Simple Good Turing
        """
        no_of_unigrams = 0

        for doc in docs:
            for sent in doc:
                no_of_unigrams += len(sent)
                for i in range(self.n):
                    tokens = list(ngrams(pad_sequence(sent, i+1, pad_left=True, pad_right=True, 
                                          left_pad_symbol='<s>', right_pad_symbol='</s>'), i+1))

                    for token in tokens:
                        if token not in self.index[i]:
                            self.index[i][token] = 1
                        else:
                            self.index[i][token] += 1
        
        for i in range(self.n - 1):
            self.index[i][('<s>',) * (i+1)] = no_of_unigrams
            self.index[i][('</s>',) * (i+1)] = no_of_unigrams
        
        if self.smooth == 'SGT':
            self.Nc = Counter(self.index[self.n-1].values())
            b, a = self.fitCurve()
            self.a = a
            self.b = b
            self.Ngrams = sum(self.index[self.n-1].values())
            self.Nc[0] = len(self.index[0])**self.n - len(self.index[self.n-1])
             
    def getNc(self, c):
        """
        Function to get number of types with count c
        """
        if self.a == None or self.b == None:
            raise Exception("Only for SGT smoothing")
        
        if c > 5:
            return pow(10, self.a + (self.b * np.log10(c)))
        else:
            return self.Nc[c]
                       
    def fitCurve(self):
        """
        Fit a linear regression curve for logN vs logC 
        """

        if self.smooth != 'SGT':
            return
        
        #Fitting logy = mlogx + c 
        
        x = list(self.Nc.keys())
        y = list(self.Nc.values())

        m, c, _, _, _ = stats.linregress(np.log10(x), np.log10(y))
    
        return m, c
    
                       
    def getTokenProbability(self, ngram):
        """
        Returns probability of the ngram token
        """
        
        if len(ngram) > self.n:
            raise Exception("Token size is larger than Modeled, Use getSentProbability")
        
        V = len(self.index[0]) + 1
        
        n = len(ngram) - 1
        
        C, C_all = 0, 0
        
        if ngram in self.index[n]:
            C = self.index[n][ngram]

        if n == 0: 
            C_all = sum([c for t, c in self.index[n].items()])
                       
        elif ngram[:-1] in self.index[n-1]:
            C_all = self.index[n-1][ngram[:-1]]
        
        if self.smooth == 'Laplace':
            return (C + self.k) / (C_all + self.k * V)
        
        elif self.smooth == 'SGT':
            C_star = (C+1) * self.getNc(C+1) / self.getNc(C)
            
            return C_star / self.Ngrams
        else:
            return C / C_all if C_all != 0 else 0 
    
    def getSentProbability(self, sent, pad_right=True):
        """
        Returns the probability of sentence
        args1: Str
        """
        sent = self.tokenize([sent])[0]

        tokens = list(ngrams(pad_sequence(sent, self.n, pad_left=True, pad_right=pad_right, 
                                          left_pad_symbol='<s>', right_pad_symbol='</s>'), self.n))
        prob = 1
        
        for token in tokens:
            prob *= self.getTokenProbability(token)
        return prob

    def getCandidateSentences(self, sent):
        """
        Returns a list of candidate sentences for spell correction
        args1: Str
        """
        candidate_sentences = [sent]
        tokenized_sent = self.tokenize([sent])[0]
        
        for pos, word in enumerate(tokenized_sent):
            for candidate_word in edit_distance1_words(word):
                
                if (candidate_word,) in self.index[0]:
                    candidate_sentence = tokenized_sent[:pos] + [candidate_word] + tokenized_sent[pos+1:]
                    candidate_sentences.append(" ".join(candidate_sentence))
    
        return candidate_sentences
    
    def autoCorrect(self, sent, alpha=0.9):
        """
        Returns a sentence with corrected spelling (atmost at one position)
        args1: str 
        """

        candidate_sentences = self.getCandidateSentences(sent)
        if len(candidate_sentences) == 1:
            return candidate_sentences[0]
        
        probs = [(1 - alpha) / (len(candidate_sentences) - 1) for _ in candidate_sentences]
        probs[0] = alpha
        
        for i, sent in enumerate(candidate_sentences):
            probs[i] *= self.getSentProbability(sent, pad_right=False)


        return candidate_sentences[np.argmax(probs)]
        
    def predictNextWord(self, prev):
        """
        Returns next word given previous(context) words by sampling based on their probabilities
        args1: tuple of strs
        """
        
        prev = tuple(prev[-self.n+1:])
        candidates, probabilities = [], []
        
        for token in list(self.index[0].keys()):
            p = self.getTokenProbability(prev + token)
            if p > 0:
                candidates.append(token[0])
                probabilities.append(p)

        if not candidates:
            return '</s>'

        return str(np.random.choice(candidates, 1, p=np.array(probabilities)/sum(probabilities))[0])
    
    def generateSentence(self, prev):
        """
        Returns predicted next words of given prev(context) sentence with utmost 'm' words
        args1: str 
        """
        generated = []
        prev = self.tokenize([prev])[0]
        while not generated or (generated[-1] != '</s>' and len(generated) != self.m):
            pred = self.predictNextWord(prev)
            generated.append(pred)
            prev += (generated[-1], )

        if generated[-1] == '</s>':
            return " ".join(generated[:-1])

        return " ".join(generated)
    
    
    def perplexity(self, test):
        """
        Returns perplexity of the model computed on the test data
        args1: list
			A list of lists of lists where each sub-sub-list a sequence of tokens
			representing a sentence
        """

        logprobabilities = []
    
        for doc in test:
            for sent in doc:
                p = 0
                tokens = list(ngrams(pad_sequence(sent, self.n, pad_left=True, pad_right=True, 
                                          left_pad_symbol='<s>', right_pad_symbol='</s>'), self.n))

                for token in tokens:
                    p += np.log2(self.getTokenProbability(token))

            logprobabilities.append(p)
        cross_entropy = -np.mean(np.array(logprobabilities))
        return cross_entropy