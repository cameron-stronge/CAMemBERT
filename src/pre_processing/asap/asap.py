import pandas as pd
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
preprocess_mod_ind = currentdir.split('/').index('BERT-Projects')
preprocess_mod_path = '/'.join(currentdir.split('/')[:preprocess_mod_ind+1])
sys.path.append(preprocess_mod_path)
from src.pre_processing.utils import (nomalise_scores,normalise_params,save_split_as_csv,
                                      save_split_as_pickle,save_pickle_file,split_data)

dataset_name = 'asap'
task = 'aes'
subtask = 'regression'

# /Users/cameronstronge/BERT-Projects/datasets/originals/asap/training_set_rel3.tsv

original_file_path = f'datasets/originals/{dataset_name}/'
pre_processed_file_path = f'datasets/pre_processed/{task}/{subtask}/'

ASAP = pd.read_csv(f'{original_file_path}training_set_rel3.tsv',
      sep='\t',encoding='latin').astype(int,errors='ignore').set_index('essay_id')

columns_of_interest = ['essay_set','essay','domain1_score']

ASAP = ASAP[columns_of_interest]

ASAP['norm_scores'] = nomalise_scores(ASAP,'essay_set','domain1_score')
normalised_param = normalise_params(ASAP,'essay_set','domain1_score')

ASAP.columns = ['essay_set','essay','labels','norm_scores']

train,test,val = split_data(ASAP)

save_split_as_csv([train,test,val],pre_processed_file_path,dataset_name)
save_split_as_pickle([train,test,val],pre_processed_file_path,dataset_name)    
save_pickle_file('nomalised_params',pre_processed_file_path,normalised_param)

