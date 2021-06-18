import pickle
import pandas as pd
import numpy as np
from datasets import load_dataset


def save_split_as_csv(dfs,file_path,dataset_title):
  for df,set_type in zip(dfs,['_train','_val','_test']):
    df.to_csv(f'{file_path}csv_files/{dataset_title}{set_type}.csv')

def save_split_as_pickle(dfs,file_path,dataset_title):
  for df,set_type in zip(dfs,['_train','_val','_test']):
    df.to_pickle(f'{file_path}pickle_files/{dataset_title}{set_type}.pickle')

def save_pickle_file(title,path,variable):
    filename = f'{path}pickle_files/{title}.pickle'                      
    filehandler = open(filename, 'wb') 
    pickle.dump(variable, filehandler)
    filehandler.close()

def split_data(df,train_split=0.6):
    val_test_split = 1-((1-train_split)*0.5)
    return np.split(df.sample(frac=1), [ int(len(df)*train_split), int(len(df)*val_test_split)])

def normalise_params(df,sets,scores):
  group_by_obj=df.groupby(sets)[scores]
  min_scores = group_by_obj.min()
  ranges = group_by_obj.max() - min_scores
  return {'min_scores':min_scores.to_dict(),'ranges':ranges.to_dict()}

def nomalise_scores(df,sets,scores):
  return df.groupby(sets)[scores].apply(lambda x: (x-min(x))/(max(x)-min(x)))