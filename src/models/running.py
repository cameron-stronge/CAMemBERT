import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
preprocess_mod_ind = currentdir.split('/').index('CAMemBERT')
preprocess_mod_path = '/'.join(currentdir.split('/')[:preprocess_mod_ind+1])
sys.path.append(preprocess_mod_path)
from transformers import AutoTokenizer,TrainingArguments
from decimal import Decimal
import json
from collections import defaultdict
from datetime import datetime

def generate_file_paths(dataset_title,task,subtask,params,abrev_name):
    file_paths = {}
    subtask_path = f'tasks/{task}/subtasks/{subtask}/'
    path_to_dataset = f'processed_data/{subtask_path}/'
    results_file_path = f'results/{subtask_path}'
    file_paths['results_file_path'] = f'results/{subtask_path}'
    file_paths['evaluation_folder'] = f'{results_file_path}evaluation_results/'
    file_paths['plots_path'] = f'{results_file_path}{dataset_title}/plots/'
    csv_files = f'{path_to_dataset}csv_files/'
    file_paths['csv_files'] = csv_files
    pickle_files = f'{path_to_dataset}pickle_files/'
    file_paths['pickle_files'] = pickle_files
    if params['dev'] == True:
        file_paths['train_file_path'] = f'{csv_files}{dataset_title}_dev_train.csv'
        file_paths['test_file_path'] = f'{csv_files}{dataset_title}_dev_test.csv'
        file_paths['val_file_path'] = f'{csv_files}{dataset_title}_dev_val.csv'
    else: 
        file_paths['train_file_path'] = f'{csv_files}{dataset_title}_train.csv'
        file_paths['test_file_path'] = f'{csv_files}{dataset_title}_test.csv'
        file_paths['val_file_path'] = f'{csv_files}{dataset_title}_val.csv'
    file_paths['original_file_path'] = f'datasets/originals/{dataset_title}/'
    if params['dev'] == True:
        file_paths['full_model_name'] = f'{abrev_name}_dev_set_{str(params["full_epochs"])}epochs_{str(params["batch_size"])}bs_{Decimal(str(params["lr"])):1E}lr_{str(params["frozen_layers"])}layers_froze'
    else:
        file_paths['full_model_name'] = f'{abrev_name}_{str(params["full_epochs"])}epochs_{str(params["batch_size"])}bs_{Decimal(str(params["lr"])):1E}lr_{str(params["frozen_layers"])}layers_froze'
    file_paths['output_path'] = f'{results_file_path}/training_output/{dataset_title}/{file_paths["full_model_name"]}'
    file_paths['evaluation_file_this_model'] = f"{file_paths['evaluation_folder']}{file_paths['full_model_name']}.json"
    file_paths['evaluation_file_subtask'] = f"{results_file_path}evaluation_file.json"
    return file_paths

def generate_parameters(es=2,lr=1e-5,bs=16,epochs=5,frozen_layers = 1,metric_for_best_model = None,dev =False,max_n_tokens=512):
    params = {}
    params['early_stopping'] = es
    params['lr'] = lr
    params['batch_size'] = bs
    params['full_epochs'] = epochs
    params['frozen_layers'] = frozen_layers
    params['metric_for_best_model'] = metric_for_best_model
    params['dev'] = dev
    params['max_n_tokens'] = max_n_tokens
    return params

def generate_eval_data(history,params,task,subtask,pre_trained_model_name):
    eval_data={}
    eval_data['pre_trained_model_name'] = pre_trained_model_name
    eval_data['task'] = task
    eval_data['subtask'] = subtask
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    eval_data['time'] = dt_string
    return {**history,**params,**eval_data}

def base_training_args(params,file_paths):
    return TrainingArguments(
        learning_rate=params['lr'],
        num_train_epochs=params['full_epochs'],
        per_device_train_batch_size=params['batch_size'],
        per_device_eval_batch_size=params['batch_size'],
        output_dir=file_paths['output_path'],
        overwrite_output_dir=True,
        evaluation_strategy='epoch',
        remove_unused_columns=False,
        metric_for_best_model=params['metric_for_best_model'],
        greater_is_better=True,
        load_best_model_at_end = True,
    ) 

def update_evaluation_results_for_this_model(eval_file_path,eval_data):
    f = open(eval_file_path,'w')
    json.dump(eval_data, f)
    f.close()

def update_evaluation_results_for_subtask(file_paths):
    if os.path.isfile(file_paths['evaluation_file_subtask']):
        f = open(file_paths['evaluation_file_subtask'],)
        eval_subtask = json.load(f)
        f.close()
        f = open(file_paths['evaluation_file_this_model'],)
        eval_data = json.load(f)
        f.close()
        cur_length = len(list(eval_subtask.values())[0])
        for key in eval_subtask.keys():
            if key not in eval_data.keys():
                eval_data[key] = None
        for key in eval_data.keys():
            if key not in eval_subtask.keys():
                eval_subtask[key] = [None]*cur_length
        for k,v in eval_data.items():
            eval_subtask[k].append(v)
        update_evaluation_results_for_this_model(file_paths['evaluation_file_subtask'],eval_subtask)
    else:
        f = open(file_paths['evaluation_file_this_model'],)
        eval_data = json.load(f)
        f.close()
        eval_subtask = defaultdict(list)
        for k,v in eval_data.items():
            eval_subtask[k].append(v)
        update_evaluation_results_for_this_model(file_paths['evaluation_file_subtask'],eval_subtask)
