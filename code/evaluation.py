from util import *

# Add your import statements here




class Evaluation():
	def __init__(self, query_scores_req=False):
		"""
		If query_sccores_req is True, all the meanMetrics will output scores of each query as a list 
		"""
		self.query_scores_req = query_scores_req
		

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here

		TP = len(set(query_doc_IDs_ordered[:k]) & set(true_doc_IDs))

		precision = TP / k
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here
		pr_sum = 0
		queryPrecisions = []
		for i, query_id in enumerate(query_ids):

			true_doc_IDs = get_true_doc_IDs(qrels, query_id)
			pr = self.queryPrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			pr_sum += pr
			queryPrecisions.append(pr)

		meanPrecision = pr_sum / len(query_ids)		
		if self.query_scores_req:
			return queryPrecisions
		else:
			return meanPrecision


	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		
		TP = len(set(query_doc_IDs_ordered[:k]) & set(true_doc_IDs))

		recall = TP / len(true_doc_IDs)

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here
		recall_sum = 0
		queryRecalls = []
		for i, query_id in enumerate(query_ids):

			true_doc_IDs = get_true_doc_IDs(qrels, query_id)
			recall = self.queryRecall(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			recall_sum += recall
			queryRecalls.append(recall)
	
		meanRecall = recall_sum / len(query_ids)		
	
		if self.query_scores_req:
			return queryRecalls
		else:
			return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		
		if (precision + recall) == 0:
			fscore = 0
		else:
			fscore = (2 * precision * recall) / (precision + recall)
		
		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		F_sum = 0
		queryFscores = []
		for i, query_id in enumerate(query_ids):

			true_doc_IDs = get_true_doc_IDs(qrels, query_id)
			Fscore = self.queryFscore(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			F_sum += Fscore
			queryFscores.append(Fscore)

		meanFscore = F_sum / len(query_ids)	

		if self.query_scores_req:
			return queryFscores
		else:
			return meanFscore


	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""
		#Assuming true_doc_IDs is a dict {doc_ID : position, ..}; (position is from qrels.) 

		nDCG = -1

		#Fill in code here
		relevance_scores = [0] * len(query_doc_IDs_ordered)

		for i, doc_ID in enumerate(query_doc_IDs_ordered):
			if doc_ID in true_doc_IDs:
				relevance_scores[i] = 5 - true_doc_IDs[doc_ID]
		
		relevance_scores_sorted = list(sorted(relevance_scores[:], reverse=True))

		DCG = 0
		IDCG = 0

		for i in range(k):
			DCG += relevance_scores[i] / np.log2(i + 2)
			IDCG += relevance_scores_sorted[i] / np.log2(i + 2)
		
		if DCG == 0:
			nDCG = 0
		else:
			nDCG = DCG / IDCG
		
		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here
		nDCG_sum = 0
		queryNDCGs = []
		for i, query_id in enumerate(query_ids):

			true_doc_IDs = get_true_doc_IDs(qrels, query_id, rel_required=True)

			nDCG = self.queryNDCG(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			nDCG_sum += nDCG
			queryNDCGs.append(nDCG)

		meanNDCG = nDCG_sum / len(query_ids)
		if self.query_scores_req:
			return queryNDCGs
		else:
			return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here
		precision_sum = 0
		rel = 0
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				rel += 1
				precision_sum += self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i + 1)

		if rel == 0:
			avgPrecision = 0
		else:
			avgPrecision = precision_sum / rel

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		#Fill in code here
		AP_sum = 0
		queryAPs = []
		for i, query_id in enumerate(query_ids):

			true_doc_IDs = get_true_doc_IDs(qrels, query_id)
			AP = self.queryAveragePrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			AP_sum += AP
			queryAPs.append(AP)

		meanAveragePrecision = AP_sum / len(query_ids)
		if self.query_scores_req:
			return queryAPs
		else:
			return meanAveragePrecision

