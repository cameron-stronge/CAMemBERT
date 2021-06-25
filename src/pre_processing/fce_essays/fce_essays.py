import pandas as pd
import glob
from bs4 import BeautifulSoup
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
preprocess_mod_ind = currentdir.split('/').index('CAMemBERT')
preprocess_mod_path = '/'.join(currentdir.split('/')[:preprocess_mod_ind+1])
sys.path.append(preprocess_mod_path)
from src.utils.preprocessing import save_split_as_csv,save_split_as_pickle,split_data

dataset_name = 'fce_essays'
task = 'aes'
subtask = 'regression'

original_file_path = f'datasets/originals/{dataset_name}/'
pre_processed_file_path = f'processed_data/tasks/{task}/subtasks/{subtask}/'

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
    df = pd.DataFrame({'essays':essays,'labels':scores})
    df['essays'] = df.essays.str.replace('\n',' ').str.replace('\t',' ').str.replace('\s',' ').str.replace("\\","")
    return df

fce_data = build_dataset(original_file_path)

fce_data['essay_set'] = 1
min_score = fce_data['labels'].min()
score_range = fce_data['labels'].max() - min_score
fce_data['norm_scores'] = (fce_data['labels']-min_score)/score_range

train,test,val = split_data(fce_data)

save_split_as_csv([train,test,val],pre_processed_file_path,dataset_name)
save_split_as_pickle([train,test,val],pre_processed_file_path,dataset_name)