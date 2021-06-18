import pandas as pd
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
preprocess_mod_ind = currentdir.split('/').index('BERT-Projects')
preprocess_mod_path = '/'.join(currentdir.split('/')[:preprocess_mod_ind+1])
sys.path.append(preprocess_mod_path)
from src.pre_processing.utils import save_split_as_csv,save_split_as_pickle

dataset_name = 'fce_grammar'
task = 'ged'
subtask1 = 'seq_2_seq'
subtask2 = 'seq_class'

original_file_path = f'datasets/originals/{dataset_name}/'
pre_processed_file_path_s2s = f'datasets/pre_processed/{task}/{subtask1}/'
pre_processed_file_path_sc = f'datasets/pre_processed/{task}/{subtask2}/'

train = pd.read_csv(f'{original_file_path}fce-public.train.original.tsv',sep='\s',names=['word','label'])
val = pd.read_csv(f'{original_file_path}fce-public.dev.original.tsv',sep='\s',names=['word','label'])
test = pd.read_csv(f'{original_file_path}fce-public.test.original.tsv',sep='\s',names=['word','label'])


def convert_words_to_sentances(df):
    l_1_words,l_2_words,l_1_labels,l_2_labels = [],[],[],[]
    for word,label in df.values:
        if word=='.' or word=='!' or word=='?':
            l_2_words.append(word)
            l_2_labels.append(label)
            l_1_words.append(' '.join(l_2_words))
            if 'i' in l_2_labels:
                l_1_labels.append(1)
            else:
                l_1_labels.append(0)
            l_2_words,l_2_labels = [],[]
        else:
            word = word.replace('ö','').replace('ô','')
            l_2_words.append(word)
            l_2_labels.append(label)

    return pd.DataFrame({'tokens':l_1_words,'labels':l_1_labels})

seq_class_train,seq_class_val,seq_class_test = [convert_words_to_sentances(df) for df in [train,test,val]]


save_split_as_csv([train,test,val],pre_processed_file_path_s2s,dataset_name)
save_split_as_pickle([train,test,val],pre_processed_file_path_s2s,dataset_name)

save_split_as_csv([seq_class_train,seq_class_val,seq_class_test],pre_processed_file_path_sc,dataset_name)
save_split_as_pickle([seq_class_train,seq_class_val,seq_class_test],pre_processed_file_path_sc,dataset_name)