import re


class TextCleaner():

	def fromList(self, text):
		"""
		Cleaning text using regular expression by removing all characters other than alphabets

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with characters other than alphabets removed
		"""
		cleanedText = []
		for sentence in text:
			cleanedSentence = []

			for token in sentence:
				cleanedToken = re.sub('[^a-z]+', '', token.lower())
				if cleanedToken:
					cleanedSentence.append(cleanedToken)

			cleanedText.append(cleanedSentence)

		return cleanedText



	