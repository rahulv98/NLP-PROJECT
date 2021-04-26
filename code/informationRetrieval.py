from util import *

# Add your import statements here
import numpy as np



class InformationRetrieval():

	def __init__(self):
		self.index = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		index = None

		#Fill in code here

		index = {}

		for i, doc in enumerate(docs):
			for sent in doc:
				for word in sent:
					if word in index:
						if docIDs[i] in index[word]:
							index[word][docIDs[i]] += 1
						else:
							index[word][docIDs[i]] = 1
					else:
						index[word] = {docIDs[i] : 1} 

		self.index = index
		self.D = len(docs)
		self.Vocab = list(index.keys())

	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		#Fill in code here

		#Initialising 
		TFIDF_docs = np.zeros((len(self.Vocab), self.D))
		IDF = np.zeros(len(self.Vocab))
		TFIDF_queries = np.zeros((len(self.Vocab), len(queries)))
		
		#Calculating TFIDF for docs		
		for i, word in enumerate(self.Vocab):			
			DF = len(self.index[word])
			IDF[i] = np.log(self.D / DF)

			for docID, count in self.index[word].items():
				TFIDF_docs[i][docID - 1] = count * IDF[i]   # docID starts at 1


		#Calculating TFIDF for Queries
		for i, query in enumerate(queries):
			for sent in query:
				for word in sent:
					try :
						word_idx = self.Vocab.index(word)
						TFIDF_queries[word_idx][i] += 1
					except:
						pass

			TFIDF_queries[:,i] = TFIDF_queries[:,i] * IDF

		#Normalising
		TFIDF_docs /= np.linalg.norm(TFIDF_docs, axis=0, keepdims=True) + 1e-4
		TFIDF_queries /= np.linalg.norm(TFIDF_queries, axis=0, keepdims=True) + 1e-4

		#Similarity Calculation
		similarity = np.matmul(TFIDF_queries.T, TFIDF_docs)

		#Ranking
		doc_IDs_ordered = (np.argsort(-similarity, axis=1) + 1).tolist()

		return doc_IDs_ordered




