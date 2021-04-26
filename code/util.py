# Add your import statements here
import numpy as np


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