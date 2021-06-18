import pandas as pd
import glob
from bs4 import BeautifulSoup
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
preprocess_mod_ind = currentdir.split('/').index('BERT-Projects')
preprocess_mod_path = '/'.join(currentdir.split('/')[:preprocess_mod_ind+1])
sys.path.append(preprocess_mod_path)
from src.pre_processing.utils import save_split_as_csv,save_split_as_pickle,split_data

dataset_name = 'fce_essays'
task = 'aes'
subtask = 'regression'

original_file_path = f'datasets/originals/{dataset_name}/'
pre_processed_file_path = f'datasets/pre_processed/{task}/{subtask}/'

def build_dataset(original_file_path):
    essays = []
    scores = []
    languages = []
    for doc in glob.glob(f'{original_file_path}*/*'):
        contents = open(doc)
        soup = BeautifulSoup(contents,'lxml')
        cur_score = soup.find('score').text
        cur_lang = soup.find('language').text
        cur_essays = []
        for essay in soup.find_all('coded_answer'):
            for c in essay.select('c'):
                c.extract()
            cur_essays.append(essay.text)
        cur_essay = ' [SEP] '.join(cur_essays)
        essays.append(cur_essay)
        scores.append(cur_score)
        languages.append(cur_lang)
    df = pd.DataFrame({'essays':essays,'scores':scores})
    df['essays'] = df.essays.str.replace('\n',' ').str.replace('\t',' ').str.replace('\s',' ').str.replace("\\","")
    return df

fce_data = build_dataset(original_file_path)

train,test,val = split_data(fce_data)

save_split_as_csv([train,test,val],pre_processed_file_path,dataset_name)
save_split_as_pickle([train,test,val],pre_processed_file_path,dataset_name)