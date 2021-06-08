from util import *

# Add your import statements here
from nltk.corpus import stopwords
import string


class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""
		#Fill in code here

		stopWords = set(stopwords.words('english'))

		stopwordRemovedText = []

		for sentence in text:
			stopwordRemovedSentence = []

			for token in sentence:
				if token not in stopWords:
					stopwordRemovedSentence.append(token)

			stopwordRemovedText.append(stopwordRemovedSentence)

		return stopwordRemovedText




	