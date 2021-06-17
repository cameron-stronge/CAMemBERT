import pandas as pd
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
preprocess_mod_ind = currentdir.split('/').index('datasets')
preprocess_mod_path = '/'.join(currentdir.split('/')[:preprocess_mod_ind+1])
sys.path.append(preprocess_mod_path)
from PreProcessing import save_data_splits,save_hg_dataset
from sklearn.model_selection import train_test_split

fce_train = pd.read_csv(f'{currentdir}/original/fce-public.train.original.tsv',sep='\s',names=['word','label'])
fce_val = pd.read_csv(f'{currentdir}/original/fce-public.dev.original.tsv',sep='\s',names=['word','label'])
fce_test = pd.read_csv(f'{currentdir}/original/fce-public.test.original.tsv',sep='\s',names=['word','label'])


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

seq_class_fce_train,seq_class_fce_val,seq_class_fce_test = [convert_words_to_sentances(df) for df in [fce_train,fce_val,fce_test]]

save_data_splits(dataset_title='fce_sq2sq',path=currentdir,dfs=[fce_train, fce_val,fce_test])
save_hg_dataset(dataset_title='fce_sq2sq',path=currentdir)

save_data_splits(dataset_title='fce_sq_cls',path=currentdir,dfs=[seq_class_fce_train,seq_class_fce_val,seq_class_fce_test])
save_hg_dataset(dataset_title='fce_sq_cls',path=currentdir)
