import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
preprocess_mod_ind = currentdir.split('/').index('datasets')
preprocess_mod_path = '/'.join(currentdir.split('/')[:preprocess_mod_ind+1])
sys.path.append(preprocess_mod_path)
from PreProcessing import save_data_splits,save_hg_dataset,split_data


def min_scores_and_ranges(df):
  group_by_obj=df.groupby('essay_set')['domain1_score']
  min_scores = group_by_obj.min()
  ranges = group_by_obj.max() - min_scores
  return {'min_scores':min_scores.to_dict(),'ranges':ranges.to_dict()}

def nomalise_scores(df):
  return df.groupby('essay_set')['domain1_score'].apply(lambda x: (x-min(x))/(max(x)-min(x)))


ASAP = pd.read_csv(f'{currentdir}/original/training_set_rel3.tsv',
      sep='\t',encoding='latin').astype(int,errors='ignore').set_index('essay_id')

columns_of_interest = ['essay_set','essay','domain1_score']

ASAP= ASAP[columns_of_interest]

ASAP['norm_scores'] = nomalise_scores(ASAP)

train,test,val = split_data(ASAP)

min_max_dic = min_scores_and_ranges(ASAP)
filename = f'{currentdir}/PreProcessed/ASAP_min_max_dic.pickle'                      
filehandler = open(filename, 'wb') 
pickle.dump(min_max_dic, filehandler)
filehandler.close()

save_data_splits(dataset_title='asap',path=currentdir,dfs=[train,test,val])
save_hg_dataset(dataset_title='asap',path=currentdir)
