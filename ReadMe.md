ReadMe.md

This Repository is associated with my final year project: "UTILISING AUXILIARY TASKS TO ENHANCE AUTOMATED ASSESSMENT THROUGH MULTI-TASK LEARNING".

The repositroy structure relies on 4 modules to run a model. trainer_f,models_f,preprocessing_f,trainer_utils.

Examples of how these modeules are to be used in the scripts starting with runing_examperiments

Currently the required data for the FCE dataset is stored in the datas folder. The data for the Conll2003 dataset is downloaded from the huggingface datasets library

Pre-processing data is not strored but instead is formed as needed. As the datasets required are dependent on the model that needs to be run.

Additionally all results for experiments referenced in the report can be accessed from the results/eval and results/test folders.
There is also a results_script which puts all results from the raw_results folder into a .csv format.

In there results folders also exists plots of losses for each model run inaddition to a plot of the models predictions vs the true labels.

There also folders entitled adapters and R2_BERT these are perifary experiments. 
R2_BERT refers to a state of the art model which was used to compare the results of this experiment to. 
Adapters offer an alternative lighterweight model with an additional alternative to multi-models without where the models are optimized seperatley and then trained together.
The results for these experiments can be found in the results/R2_BERT and results/adapters. The final line of each csv file refers to the model performance on the test datasets.