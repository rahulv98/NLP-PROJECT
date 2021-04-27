# Add your import statements here
import numpy as np
from nltk.util import ngrams

# Add any utility functions here

def get_true_doc_IDs(qrels, query_id, rel_required=False):

    if rel_required:
        true_doc_IDs = {}
    else:
        true_doc_IDs = []

    for judgement in qrels:
        if int(judgement['query_num']) == query_id:
            if rel_required:
                true_doc_IDs[int(judgement['id'])] = int(judgement['position'])
            else:
                true_doc_IDs.append(int(judgement['id']))

    return true_doc_IDs

def nGramConverter(docs, N):
    '''
    arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
    arg2 : int
            n in 'n-grams'
    '''
    docs_conv = []
    for doc in docs:
        doc_conv = []
        for sent in doc:
            if len(sent) >= N:
                doc_conv.append(list(ngrams(sent, N)))

        docs_conv.append(doc_conv)
    
    return docs_conv