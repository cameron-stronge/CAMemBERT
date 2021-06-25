import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
preprocess_mod_ind = currentdir.split('/').index('CAMemBERT')
preprocess_mod_path = '/'.join(currentdir.split('/')[:preprocess_mod_ind+1])
sys.path.append(preprocess_mod_path)
from transformers import AutoTokenizer,EarlyStoppingCallback,TrainingArguments
from datasets import load_dataset,load_metric
from src.utils.preprocessing import read_pickle_file
from src.models.adaptor_models import TaggingModelAdaptors
from src.models.trainers import Seq2SeqTrainer
from src.models.running import *
from sklearn.metrics import r2_score,cohen_kappa_score
import torch
import torch.nn.functional as F
from datasets import load_metric,load_dataset
from sklearn.metrics import f1_score,accuracy_score
import ast
import numpy as np
import pandas as pd

def encode_data(data):
    encoded = tokenizer([" ".join(doc) for doc in data[tokenize_col]], pad_to_max_length=True, padding="max_length",
                        max_length=max_n_tokens, truncation=True, add_special_tokens=True)
    return (encoded)

def encode_labels(example):
    r_tags = []
    count = 0
    token2word = []
    tokens = ast.literal_eval(example[tokenize_col])
    labels = ast.literal_eval(example["labels"])

    for index, token in enumerate(tokenizer.tokenize(" ".join(tokens))):

        if token.startswith("##") or (token in tokens[index - count - 1].lower() and index - count - 1 >= 0):
            # if the token is part of a larger token and not the first we need to differ 
            # if it is a B (beginning) label the next one needs to ba assigned a I (intermediate) label
            # otherwise they can be labeled the same
            r_tags.append(r_tags[-1])
            count += 1
        else:
            
            r_tags.append(labels[index - count])

        token2word.append(index - count)


    r_tags = torch.tensor(r_tags)
    labels = {}
    # Pad token to maximum length for using batches
    labels["labels"] = F.pad(r_tags, pad=(1, max_n_tokens - r_tags.shape[0]), mode='constant', value=0)
    # Truncate if the document is too long
    labels["labels"] = labels["labels"][:max_n_tokens]

    return labels

def compute_metrics(p):
    logits, labels = p.predictions,p.label_ids
    logits = logits.flatten()
    labels = labels.flatten()
    metrics_dic = {}
    metrics_dic['f1'] = f1_score(logits, labels)
    metrics_dic['accuracy'] = accuracy_score(logits,labels)
    return metrics_dic

es = 2
lr = 1e-5
bs = 16
epochs = 5
frozen_layers = 0
metric_for_best_model = 'f1'
dev = True

tokenize_col='tokens'
max_n_tokens=128
extra_cols_for_dataset= []

params = generate_parameters(es,lr,bs,epochs,frozen_layers,metric_for_best_model,dev,max_n_tokens)

pre_trained_model_name = 'bert-base-uncased'
dataset_title = 'fce_grammar'
task = 'ged'
subtask = 'seq_2_seq'
abrev_name = f'tagging_bert_weighted_loss_func_adaptors_{dataset_title}'

file_paths = generate_file_paths(dataset_title,task,subtask,params,abrev_name)

dataset_dic = {'train':file_paths['train_file_path'],'test':file_paths['test_file_path'],'val':file_paths['val_file_path']}

dataset = load_dataset('csv', data_files=dataset_dic)

tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)

dataset = dataset.map(encode_labels)
dataset = dataset.map(encode_data, batched=True, batch_size=16)

dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
# model.model(**test)

train_df_original = pd.read_csv(f'{file_paths["original_file_path"]}fce-public.train.original.tsv',sep='\s',names=['word','label'])
n_0, n_1 = train_df_original['label'].value_counts()
w_0 = (n_0 + n_1) / (2.0 * n_0)
w_1 = (n_0 + n_1) / (2.0 * n_1)
class_weights = [w_0,w_1]
class_weights_tensor=torch.FloatTensor([w_0, w_1]).cuda()

labels=['c','i']
num_labels = len(labels)
adapter_names=f'{file_paths["full_model_name"]}_{dataset_title}'
model = TaggingModelAdaptors(pre_trained_model_name,labels,adapter_names,class_weights_tensor)
model.activate_adapters()
model = model.model

training_args = base_training_args(params,file_paths)

callbacks = [EarlyStoppingCallback(es)]

trainer = Seq2SeqTrainer(model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    compute_metrics=compute_metrics,
    callbacks=callbacks,)

trainer.train()

print()
print('__________test set results__________')
trainer.evaluate(dataset['test'],testing=True)
hist = trainer.state.log_history[-1]
eval_data = generate_eval_data(hist,params,task,subtask,pre_trained_model_name)
update_evaluation_results_for_this_model(f'{file_paths["evaluation_file_this_model"]}',eval_data)
update_evaluation_results_for_subtask(file_paths)

print('_________plotting history__________')
trainer.plot_history(file_paths['plots_path'],f'{file_paths["full_model_name"]}')