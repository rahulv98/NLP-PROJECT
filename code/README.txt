This folder contains the implementation of our project -> search engine with VSM + LSA + ngram indexing + auto complete query + spell corrector.

main.py - The main module that contains the outline of the search engine.
util.py - Has additional utility functions

sentenceSegmentation.py - Has a class module for sentence segmentation
tokenization.py - Has a class module for tokenization
textCleaner.py - Has a class module for cleaning tokenized text
inflectionReduction.py - Has a class module for Lemmatization
stopwordRemoval.py - Has a class module for removing stop words
informationRetrieval.py - Has a module for indexing and ranking documents with LSA and ngram-indexing
evaluation.py - Has all the methods for different metric evaluations

plots.py - has a method to plot all the distribution comparison plots along with hypothesis testing attached in report.
LanguageModel.py - Has a class module with Language model implementation along with auto complete and spell correct methods

To get the comparision plots as in report, run plots.py

To test the code, run main.py with appropriate arguments
Usage:  main.py [-h] [-dataset DATASET] [-out_folder OUT_FOLDER] [-segmenter SEGMENTER] [-tokenizer TOKENIZER] [-LSA] [-base1] [-K K]
               [-IR_n IR_N] [-tune_lsa] [-custom] [-autocomplete] [-spell_check] [-LM_k LM_K] [-LM_smooth LM_SMOOTH] [-LM_n LM_N]
               [-LM_m LM_M] [-perplexity]

When the -custom flag is passed, the system will take a query from the user as input. When the flag is not passed, all the queries in the Cranfield dataset are considered, for example:
> python main.py -custom
> Enter query below
> Papers on Aerodynamics

When the -spell_check is passed along with -custom flag and language model parameters, the system will take a query from the user as input. and suggests corrected query and asks again for input and after the input retrieves documents, for example:
>python main.py -LM_n 3 -custom -spell_check -LM_smooth SGT
>Enter query below
>hat analytical solutions
>proposed corrected query:-  what analytical solutions
>Enter query below
>what analytical solutions
(Can input any query we want)

When the -autocomplete is passed along with -custom flag and language model parameters, the system will take a query from the user as input. and suggests complet query and asks again for input and after the input retrieves documents, for example:
>python main.py -LM_n 3 -custom -autocomplete -LM_smooth SGT
>Enter query below:
>paper on shear buckling
>proposed completed query:- paper on shear buckling of thin cylindrical shells
>Enter query below
>paper on shear buckling of thin cylindrical shells

When the -tune_lsa flag is passed along with -LSA flag, the code plots a k vs nDCG@5 in two scales

When the -base1 flag is passed, the search engine will not use textCleaner(this gives the results of VSM-base1 model as stated in report)

When the -perplexity flag is passed along with LanguageModel parameters, then the perplexity of the model on Cranfield dataset is printed.
