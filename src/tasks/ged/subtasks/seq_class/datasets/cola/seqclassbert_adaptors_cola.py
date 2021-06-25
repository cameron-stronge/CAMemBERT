import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
preprocess_mod_ind = currentdir.split('/').index('CAMemBERT')
preprocess_mod_path = '/'.join(currentdir.split('/')[:preprocess_mod_ind+1])
sys.path.append(preprocess_mod_path)
from transformers import AutoTokenizer,EarlyStoppingCallback,TrainingArguments,AutoModelWithHeads,AutoConfig
from datasets import load_dataset,load_metric
from src.models.trainers import BaseTrainer
from src.models.running import *
from sklearn.metrics import f1_score,accuracy_score
import torch
import numpy as np

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch[tokenize_col], max_length=max_n_tokens, truncation=True, padding="max_length")

def compute_metrics(p):
    logits, labels = np.argmax(p.predictions, axis=1),p.label_ids
    logits = logits.flatten()
    labels = labels.flatten()
    metrics_dic = {}
    metrics_dic['f1'] = f1_score(logits, labels)
    metrics_dic['accuracy'] = accuracy_score(logits,labels)
    return metrics_dic

es = 2
lr = 4e-5
bs = 16
epochs = 5
frozen_layers = 1
metric_for_best_model = 'f1'
dev = False

tokenize_col='sentences'
max_n_tokens=64
extra_cols_for_dataset=[]

params = generate_parameters(es,lr,bs,epochs,frozen_layers,metric_for_best_model,dev,max_n_tokens)

pre_trained_model_name = 'bert-base-uncased'
dataset_title = 'cola'
task = 'ged'
subtask = 'seq_class'
abrev_name = f'seq_class_bert_adaptors_{dataset_title}'

file_paths = generate_file_paths(dataset_title,task,subtask,params,abrev_name)

dataset_dic = {'train':file_paths['train_file_path'].replace('_dev',''),'test':file_paths['test_file_path'].replace('_dev',''),'val':file_paths['val_file_path'].replace('_dev','')}

dataset = load_dataset('csv', data_files=dataset_dic)
tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
dataset = dataset.map(encode_batch, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask","labels"]+extra_cols_for_dataset)

config = AutoConfig.from_pretrained(
    pre_trained_model_name,
    num_labels=2,
)
model = AutoModelWithHeads.from_pretrained(
    pre_trained_model_name,
    config=config,
)

adapter_names=f'{file_paths["full_model_name"]}_{dataset_title}'
# Add a new adapter
model.add_adapter(adapter_names)
# Add a matching classification head
model.add_classification_head(
    adapter_names,
    num_labels=2,
  )
# Activate the adapter
model.train_adapter(adapter_names)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

training_args = base_training_args(params,file_paths)

callbacks = [EarlyStoppingCallback(es)]

trainer = BaseTrainer(model=model,
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