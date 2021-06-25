# CAMemBERT

I know its a cheesey name

This is a repository for my masters individual project. The goal of the project is to use a pre-trained model (such as BERT) to build a language model in the language learner domain (e.g a native spanish speaker learning english). The language model will be assessed on a number of down-stream tasks.

The general structure at present is  : tasks/task_folders/subtasks/subtask_folders/dataset_folders.

So far this has been implemented for :

- src/tasks folder where the final dataset_folder contains the sript (as well as accompanying ipython notebook) which can be run to train a model on a given dataset. Every dataset contains a model built using a traditional huggingface model and then an equivalent model build using adapters. Also to save on time most models have been bbuilt using developement sets which are 10% the size of the origing train, test and val datasets.

- results/tasks folder where the subtask_folders contains a json and csv file for each of the experiments run for that specific subtask. In addition in the final dataset_folders there are plots and model outputs for each of the experiments.

- preprocessed_data folder where the final dataset_folders contains csv for the train, test and val as well as some dev sets. (There is at present a discrepancy between the code for the fce_essays dataset code and data, so use the dev set if you are looking to run this code: this can be done by setting dev to true in script to train mode).

In addition in the src there are a number of baseline classes in the models folders, trainer classes for training a model, scripts for pre-processing data, some helper functions in the utils folder.

Expect a clean up sometime soon.