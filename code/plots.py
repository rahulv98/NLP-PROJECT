from evaluation import Evaluation

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy.stats import ttest_rel

def compareDistPlot(scores1, scores2, model1_name, model2_name, metric_name, args):
    """
    Generates distribution plot for scores of 2 models and calculates it's p-value 
    """
    tstat, pval = ttest_rel(scores1, scores2)
    plt.figure(dpi= 160)
    sns.kdeplot(scores1, color="dodgerblue", label=model1_name)
    sns.kdeplot(scores2, color="red", label=model2_name)
    plt.legend()
    plt.xlabel(metric_name)
    plt.text(0.30, 0.13, f't-statistic = {round(tstat, 2)}')
    plt.text(0.33, 0.05, f'p-value = {round(pval, 3)}')
    plt.title(f'{metric_name} comparision distribution between {model1_name} and {model2_name}')
    plt.savefig(args.out_folder + f'{model1_name}_{model2_name}_{metric_name}.png')

def generateAllComparePlots(model1_preds_path, model2_preds_path, args):
    model1_preds = json.load(open(model1_preds_path, 'r'))[:]
    model2_preds = json.load(open(model2_preds_path, 'r'))[:]
    
    model1_name = model1_preds_path.split("/")[-1][6:-4]
    model2_name = model2_preds_path.split("/")[-1][6:-4]

    evaluator = Evaluation(query_scores_req=True)

    queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
    query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
    qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

    metricList = [('nDCG', evaluator.meanNDCG), 
                  ('F-Score', evaluator.meanFscore), 
                  ('Precision', evaluator.meanPrecision), 
                  ('Recall', evaluator.meanRecall), 
                  ('AvgPrecision', evaluator.meanAveragePrecision)]

    for metric_name, scorer in metricList:
        scores1 = scorer(model1_preds, query_ids, qrels, 5)
        scores2 = scorer(model2_preds, query_ids, qrels, 5)
        
        compareDistPlot(scores1, scores2, model1_name, model2_name, metric_name, args)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='plots.py')

    # Tunable parameters as external arguments
    parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
    parser.add_argument('-custom', action = "store_true", 
						help = "Plot between model1 and model2")
    parser.add_argument('-out_folder', default = "output/", 
						help = "Path to output folder")
    parser.add_argument('-model1_path', default=None,
                        help = 'Path to predictions from model1; to compare with model2')
    parser.add_argument('-model2_path', default=None,
                        help = 'Path to predictions from model2; to compare with model1')


    model1_path = 'output/preds_base1.txt'
    model2_path = 'output/preds_base2.txt'
    model3_path = 'output/preds_2_gram.txt'
    model4_path = 'output/preds_3_gram.txt'
    model5_path = 'output/preds_1_gram_LSA_410.txt'
    model6_path = 'output/preds_2_gram_LSA_390.txt'
    # Parse the input arguments
    args = parser.parse_args()

    if args.custom:
        generateAllComparePlots(args.model1_path, args.model2_path, args)

    else:
        generateAllComparePlots(model1_path, model2_path, args)
        generateAllComparePlots(model2_path, model3_path, args)
        generateAllComparePlots(model2_path, model4_path, args)
        generateAllComparePlots(model2_path, model5_path, args)
        generateAllComparePlots(model2_path, model6_path, args)
        
