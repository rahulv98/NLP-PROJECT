from util import *

# Add your import statements here
import nltk.data

class SentenceSegmentation():

    def naive(self, text):
        """
        Sentence Segmentation using a Naive Approach

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each string is a single sentence
        """

        punct = ['.', '?', '!']

        segmentedText = []
        currSegment = ""

        for char in text:
            currSegment += char

            #Break the setence if a punctuation occurs
            if char in punct:
                segmentedText.append(currSegment)
                currSegment = ""

        #Handling last segment
        if currSegment:
            segmentedText.append(currSegment)

        return segmentedText



    def punkt(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each strin is a single sentence
        """

        punktTokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
        segmentedText = punktTokenizer.tokenize(text)

        return segmentedText