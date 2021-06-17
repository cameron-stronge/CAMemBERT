import pandas as pd
import os, sys
import glob
from bs4 import BeautifulSoup
currentdir = os.path.dirname(os.path.realpath(__file__))
preprocess_mod_ind = currentdir.split('/').index('datasets')
preprocess_mod_path = '/'.join(currentdir.split('/')[:preprocess_mod_ind+1])
sys.path.append(preprocess_mod_path)
from PreProcessing import save_data_splits,save_hg_dataset,split_data
from sklearn.model_selection import train_test_split

def build_dataset(currentdir):
    essays = []
    scores = []
    languages = []
    for doc in glob.glob(f'{currentdir}/original/*/*'):
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

fce_data = build_dataset(currentdir)

train,test,val = split_data(fce_data)

save_data_splits(dataset_title='fce-aes',path=currentdir,dfs=[train,test,val])
save_hg_dataset(dataset_title='fce-aes',path=currentdir)