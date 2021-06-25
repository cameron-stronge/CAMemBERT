import pandas as pd
from sklearn.model_selection import train_test_split
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
preprocess_mod_ind = currentdir.split('/').index('CAMemBERT')
preprocess_mod_path = '/'.join(currentdir.split('/')[:preprocess_mod_ind+1])
sys.path.append(preprocess_mod_path)
from src.utils.preprocessing import save_split_as_csv,save_split_as_pickle

dataset_name = 'cola'
task = 'ged'
subtask = 'seq_class'

original_file_path = f'datasets/originals/{dataset_name}/'
pre_processed_file_path = f'processed_data/tasks/{task}/subtasks/{subtask}/'

train = pd.read_csv(f'{original_file_path}in_domain_train.tsv', delimiter='\t', header=None,
                            names=['sentence_source', 'label', 'label_notes', 'sentence'])
test = pd.read_csv(f'{original_file_path}in_domain_dev.tsv', delimiter='\t', header=None,
                            names=['sentence_source', 'label', 'label_notes', 'sentence'])

columns_of_interest = ['sentence','label']

train = train[columns_of_interest]
test = test[columns_of_interest]

train.columns = ['sentences','labels']
test.columns = ['sentences','labels']

train, val = train_test_split(train, test_size=len(test))

save_split_as_csv([train,test,val],pre_processed_file_path,dataset_name)
save_split_as_pickle([train,test,val],pre_processed_file_path,dataset_name)