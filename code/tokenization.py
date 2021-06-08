from util import *

# Add your import statements here
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TreebankWordTokenizer
import re


class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		#Fill in code here
		tokenizedText = []
		for sentence in text:
			tokens = sentence.split()
			tokenizedText.append(tokens)

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
		#Fill in code here

		tokenizedText = []
		# ptb = RegexpTokenizer(r'\w+')
		ptb = TreebankWordTokenizer()
		for sentence in text:
			tokens = ptb.tokenize(sentence)
			tokenizedText.append(tokens)

		return tokenizedText