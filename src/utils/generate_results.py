import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
preprocess_mod_ind = currentdir.split('/').index('CAMemBERT')
preprocess_mod_path = '/'.join(currentdir.split('/')[:preprocess_mod_ind+1])
sys.path.append(preprocess_mod_path)
import glob
import json
import pandas as pd
for file_path in glob.glob('results/tasks/**/subtasks/**/evaluation_file.json'):
    subtask_fp = '/'.join(file_path.split('/')[:-1])
    # Opening JSON file
    f = open(file_path,)
    data = json.load(f)
    f.close()
    eval_df = pd.DataFrame(data)
    print(eval_df)
    eval_df.to_csv(subtask_fp + '/evaluation_file.csv')