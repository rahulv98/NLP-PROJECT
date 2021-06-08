from ast import parse
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation
from textCleaner import TextCleaner
from LanguageModel import NgramModel
from util import getName

from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt
import re
import numpy as np

#Fixing random seed
np.random.seed(100)

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")


class SearchEngine:

	def __init__(self, args):
		self.args = args

		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.textCleaner = TextCleaner()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()

		self.informationRetriever = InformationRetrieval(LSA=args.LSA, n=args.IR_n, K=args.K)
		self.evaluator = Evaluation()

		self.LanguageModel = NgramModel(n=args.LM_n, smooth=args.LM_smooth, k=args.LM_k, m=args.LM_m)

	def segmentSentences(self, text):
		"""
		Call the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		"""
		Call the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)

	def reduceInflection(self, text):
		"""
		Call the required stemmer/lemmatizer
		"""
		return self.inflectionReducer.reduce(text)

	def cleanText(self, text):
		"""
		Call the text cleaner
		"""
		return self.textCleaner.fromList(text)

	def removeStopwords(self, text):
		"""
		Call the required stopword remover
		"""
		return self.stopwordRemover.fromList(text)


	def preprocessQueries(self, queries, for_LM=False):
		"""
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""

		# Segment queries
		segmentedQueries = []
		for query in queries:
			segmentedQuery = self.segmentSentences(query)
			segmentedQueries.append(segmentedQuery)
		json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
		# Tokenize queries
		tokenizedQueries = []
		for query in segmentedQueries:
			tokenizedQuery = self.tokenize(query)
			if not self.args.base1:
				tokenizedQuery = self.textCleaner.fromList(tokenizedQuery)
			tokenizedQueries.append(tokenizedQuery)
		json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))

		if for_LM:
			return tokenizedQueries 

		# Stem/Lemmatize queries
		reducedQueries = []
		for query in tokenizedQueries:
			reducedQuery = self.reduceInflection(query)
			reducedQueries.append(reducedQuery)
		json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
		# Remove stopwords from queries
		stopwordRemovedQueries = []
		for query in reducedQueries:
			stopwordRemovedQuery = self.removeStopwords(query)
			stopwordRemovedQueries.append(stopwordRemovedQuery)
		json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

		preprocessedQueries = stopwordRemovedQueries
		return preprocessedQueries

	def preprocessDocs(self, docs, for_LM=False):
		"""
		Preprocess the documents
		"""
		
		# Segment docs
		segmentedDocs = []
		for doc in docs:
			segmentedDoc = self.segmentSentences(doc)
			segmentedDocs.append(segmentedDoc)
		json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
		# Tokenize docs
		tokenizedDocs = []
		for doc in segmentedDocs:
			tokenizedDoc = self.tokenize(doc)
			if not self.args.base1:
				tokenizedDoc = self.textCleaner.fromList(tokenizedDoc)
			tokenizedDocs.append(tokenizedDoc)
		json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))

		if for_LM:
			return tokenizedDocs

		# Stem/Lemmatize docs
		reducedDocs = []
		for doc in tokenizedDocs:
			reducedDoc = self.reduceInflection(doc)
			reducedDocs.append(reducedDoc)
		json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
		# Remove stopwords from docs
		stopwordRemovedDocs = []
		for doc in reducedDocs:
			stopwordRemovedDoc = self.removeStopwords(doc)
			stopwordRemovedDocs.append(stopwordRemovedDoc)
		json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

		preprocessedDocs = stopwordRemovedDocs
		return preprocessedDocs


	def evaluateDataset(self):
		"""
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""

		# Read queries
		queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
		# Process queries 
		processedQueries = self.preprocessQueries(queries)

		# Read documents
		docs_json = json.load(open(self.args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
								[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for each query
		doc_IDs_ordered = self.informationRetriever.rank(processedQueries)
		#To use for plotting and hypothesis testing
		json.dump(doc_IDs_ordered, open(self.args.out_folder + "preds_" + getName(self.args) + ".txt", 'w'))

		# Read relevance judements
		qrels = json.load(open(self.args.dataset + "cran_qrels.json", 'r'))[:]

		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
		precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
		for k in range(1, 11):
			precision = self.evaluator.meanPrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			precisions.append(precision)
			recall = self.evaluator.meanRecall(
				doc_IDs_ordered, query_ids, qrels, k)
			recalls.append(recall)
			fscore = self.evaluator.meanFscore(
				doc_IDs_ordered, query_ids, qrels, k)
			fscores.append(fscore)
			print("Precision, Recall and F-score @ " +  
				str(k) + " : " + str(precision) + ", " + str(recall) + 
				", " + str(fscore))
			MAP = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			MAPs.append(MAP)
			nDCG = self.evaluator.meanNDCG(
				doc_IDs_ordered, query_ids, qrels, k)
			nDCGs.append(nDCG)
			print("MAP, nDCG @ " +  
				str(k) + " : " + str(MAP) + ", " + str(nDCG))

		# Plot the metrics and save plot 
		plt.plot(range(1, 11), precisions, label="Precision")
		plt.plot(range(1, 11), recalls, label="Recall")
		plt.plot(range(1, 11), fscores, label="F-Score")
		plt.plot(range(1, 11), MAPs, label="MAP")
		plt.plot(range(1, 11), nDCGs, label="nDCG")
		plt.legend()
		plt.title("Evaluation Metrics -" + getName(self.args))
		plt.xlabel("k")
		plt.savefig(self.args.out_folder + getName(self.args) + "_eval_plot.png")

	def buildLanguageModel(self):

		# Read queries
		queries_json = json.load(open(self.args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
		# Process queries 
		processedQueries = self.preprocessQueries(queries, for_LM=True)

		# Read documents
		docs_json = json.load(open(self.args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
								[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs, for_LM=True)

		self.LanguageModel.buildIndex(processedDocs + processedQueries)

		if self.args.perplexity:
			print(self.LanguageModel.perplexity(processedDocs + processedQueries))


	def LSATuning(self):
		"""
		-Preporcess Docs and queries
		-Invoke the IR system with different values of latent features(K)
		-evaluate the the model on cranfield dataset with nDCG@5 metric
		-Plot K vs nDCG@5
		"""
		# Read queries
		queries_json = json.load(open(self.args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
		# Process queries 
		processedQueries = self.preprocessQueries(queries)

		# Read documents
		docs_json = json.load(open(self.args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
								[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Read relevance judements
		qrels = json.load(open(self.args.dataset + "cran_qrels.json", 'r'))[:]

		nDCGs = []
		search_ranges = [range(100, 1000, 50), range(200, 800, 100), range(300, 700, 100)] 
		for k in search_ranges[self.args.IR_n - 1]:
			print(f'Training LSA with {k} latent features for {self.args.IR_n}-gram representation')
			#Initialise an IR system
			IR = InformationRetrieval(LSA=True, K=k, n=self.args.IR_n)
			
			#Build document index
			IR.buildIndex(processedDocs, doc_ids)
			
			# Rank the documents for each query
			doc_IDs_ordered = IR.rank(processedQueries)

			nDCG = self.evaluator.meanNDCG(
				doc_IDs_ordered, query_ids, qrels, 5)

			nDCGs.append(nDCG)

		plt.plot(search_ranges[self.args.IR_n - 1], nDCGs, marker='x')
		plt.title(f"K vs nDCG@5 for {self.args.IR_n}-grams")
		plt.xlabel("K")
		plt.ylabel("nDCG@5")
		plt.savefig(self.args.out_folder + f"LSA_{self.args.IR_n}.png")
		plt.cla()

		nDCGs = []
		fine_search_ranges = [range(360, 460, 5), range(360, 460, 10), range(460, 540, 20)] 
		for k in fine_search_ranges[self.args.IR_n - 1]:
			print(f'Training LSA with {k} latent features for {self.args.IR_n}-gram representation')
			#Initialise an IR system
			IR = InformationRetrieval(LSA=True, K=k, n=self.args.IR_n)
			
			#Build document index
			IR.buildIndex(processedDocs, doc_ids)
			
			# Rank the documents for each query
			doc_IDs_ordered = IR.rank(processedQueries)

			nDCG = self.evaluator.meanNDCG(
				doc_IDs_ordered, query_ids, qrels, 5)

			nDCGs.append(nDCG)

		plt.plot(fine_search_ranges[self.args.IR_n - 1], nDCGs, marker='x')
		plt.title(f"K vs nDCG@5 for {self.args.IR_n}-grams")
		plt.xlabel("K")
		plt.ylabel("nDCG@5")
		plt.savefig(self.args.out_folder + f"LSA_{self.args.IR_n}_fine.png")

	def handleCustomQuery(self):
		"""
		Take a custom query as input and return top five relevant documents
		"""
		if self.args.autocomplete or self.args.spell_check:
			self.buildLanguageModel()

		#Get query
		print("Enter query below")
		query = input()

		#Query spell checking
		if self.args.spell_check:
			corrected_query = self.LanguageModel.autoCorrect(query)
			print("proposed corrected query:- ", corrected_query)

			print("Enter corrected query below")
			query = input()
		#Query autocomplete upto m-words
		if self.args.autocomplete:
			pred_query = self.LanguageModel.generateSentence(query)
			print("proposed completed query:- ", query + " " + pred_query)
		

			print("Enter complete query below")
			query = input()

		# Process documents
		processedQuery = self.preprocessQueries([query])[0]

		# Read documents
		docs_json = json.load(open(self.args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
							[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for the query
		doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

		# Print the IDs of first five documents
		print("\nTop five document IDs : ")
		for id_ in doc_IDs_ordered[:5]:
			print(id_)


if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [naive|ptb]")
	
	#IR system parameters
	parser.add_argument('-LSA', action = "store_true", 
						help = "Use LSA method for IR")
	parser.add_argument('-base1', action = "store_true", 
						help = "Use baseline1 model")
	parser.add_argument('-K',  default = None, type=int,
	                    help = "Rank for LSA")
	parser.add_argument('-IR_n',  default = 1, type=int,
	                    help = "Doc representation with upto ngrams")
	parser.add_argument('-tune_lsa', action="store_true",
						help = "Hyperparameter Latent features tuning for LSA method")


	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input")
	parser.add_argument('-autocomplete', action = 'store_true',
						help = "Predict complete query for inputed custom query")
	parser.add_argument('-spell_check', action='store_true',
						help = "Correct spellings of the custom query")

	#Language Model parameters
	parser.add_argument('-LM_k',  default = None, type=int,
	                    help = "k value for Laplace smoothing")
	parser.add_argument('-LM_smooth', default = None,
						help = "Language Model Smoothing Method [None|Laplace|SGT]")
	parser.add_argument('-LM_n',  default = 1, type=int,
	                    help = "n for Ngram Language Model")
	parser.add_argument('-LM_m',  default = 4, type=int,
	                    help = "Auto complete upto m words")
	parser.add_argument('-perplexity', action='store_true',
						help = "Compute perplexity on same dataset")

	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)
	print(args)
	# Either handle query from user or evaluate on the complete dataset 

	if args.perplexity:
		searchEngine.buildLanguageModel()
	if args.custom:
		searchEngine.handleCustomQuery()
	elif args.LSA and args.tune_lsa:
		searchEngine.LSATuning() #plots LSA vs k for the chosen model
	else:
		searchEngine.evaluateDataset()
