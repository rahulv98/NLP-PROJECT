# Add your import statements here
import numpy as np
from nltk.util import ngrams
import string
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

def getName(args):
    """
    Based on the args, returns a name for the model
    """
    if args.base1:
        return 'base1'
    elif args.IR_n == 1 and not args.LSA:
        return 'base2'
    elif not args.LSA:
        return f"{args.IR_n}_gram"
    else:
        return f"{args.IR_n}_gram_LSA_{args.K}"


def edit_distance1_words(word):
    """
    Returns all the 1 edit distance words possible for a given word as list. (Word need not be in vocabulary)
    """
    splits =[]
    deletes = []
    swaps = []
    replaces = []
    inserts =[]
    candidates =[]
    letters = string.ascii_lowercase
    for pos in range(len(word)+1):
        splits.append((word[:pos],word[pos:]))
    for l,r in splits:
        if r:
            deletes.append(l+r[1:])
    for l,r in splits:
        if len(r)>1:
            swaps.append(l + r[1] + r[0] + r[2:])
    for l,r in splits:
        if r:
            for c in letters:
                replaces.append( l + c + r[1:])
    for l,r in splits:
        for c in letters:
            inserts.append(l + c + r)
    candidates = set(deletes + swaps + replaces + inserts)
    return candidates
