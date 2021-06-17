import pickle
import pandas as pd
import numpy as np
from datasets import load_dataset


def save_data_splits(dataset_title,path,dfs):
  for df,set_type in zip(dfs,['_train','_val','_test']):
    df.to_pickle(f'{path}/PreProcessed/DataFrames/{dataset_title}{set_type}.pkl')
    df.to_csv(f'{path}/PreProcessed/CsvFiles/{dataset_title}{set_type}.csv')

def save_hg_dataset(dataset_title,path):
    dataset = load_dataset('csv', data_files={'train':[f'{path}/PreProcessed/CsvFiles/{dataset_title}_train.csv'],
                                                'val':[f'{path}/PreProcessed/CsvFiles/{dataset_title}_val.csv'],
                                                'test':[f'{path}/PreProcessed/CsvFiles/{dataset_title}_test.csv']})
    filename = f'{path}/PreProcessed/HGDataset/{dataset_title}_hgdataset.pickle'                      
    filehandler = open(filename, 'wb') 
    pickle.dump(dataset, filehandler)
    filehandler.close()

def split_data(df,train_split=0.6):
    val_test_split = 1-((1-train_split)*0.5)
    return np.split(df.sample(frac=1), [ int(len(df)*train_split), int(len(df)*val_test_split)])