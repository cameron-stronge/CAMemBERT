import pandas as pd
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
preprocess_mod_ind = currentdir.split('/').index('datasets')
preprocess_mod_path = '/'.join(currentdir.split('/')[:preprocess_mod_ind+1])
sys.path.append(preprocess_mod_path)
from PreProcessing import save_data_splits,save_hg_dataset
from sklearn.model_selection import train_test_split

cola_train = pd.read_csv(f'{currentdir}/original/in_domain_train.tsv', delimiter='\t', header=None,
                            names=['sentence_source', 'label', 'label_notes', 'sentence'])
cola_test = pd.read_csv(f'{currentdir}/original/in_domain_dev.tsv', delimiter='\t', header=None,
                            names=['sentence_source', 'label', 'label_notes', 'sentence'])

columns_of_interest = ['sentence','label']

cola_train = cola_train[columns_of_interest]
cola_test = cola_test[columns_of_interest]

cola_train, cola_val = train_test_split(cola_train, test_size=len(cola_test))

save_data_splits(dataset_title='Cola',path=currentdir,dfs=[cola_train, cola_val,cola_test])
save_hg_dataset(dataset_title='Cola',path=currentdir)