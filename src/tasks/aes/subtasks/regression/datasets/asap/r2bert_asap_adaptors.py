import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
preprocess_mod_ind = currentdir.split('/').index('CAMemBERT')
preprocess_mod_path = '/'.join(currentdir.split('/')[:preprocess_mod_ind+1])
sys.path.append(preprocess_mod_path)
from transformers import AutoTokenizer,EarlyStoppingCallback,TrainingArguments
from datasets import load_dataset,load_metric
from src.utils.preprocessing import read_pickle_file
from src.models.adaptor_models import R2BERTAdaptors
from src.models.trainers import R2Trainer
from src.models.running import *
from sklearn.metrics import r2_score,cohen_kappa_score
import torch

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["essay"], max_length=512, truncation=True, padding="max_length")

def compute_metrics(p):
    #####################################################
    # Here is where the error lies p.predictions returns only 30 
    # predictions for the training arguments and parameters set below
    logits, labels = p.predictions,p.label_ids
    metrics_dic = metric.compute(predictions=logits, references=labels)
    metrics_dic['cohen_kappa'] = cohen_kappa_score(logits,labels)
    metrics_dic['pearsonr'] = r2_score(labels, logits)
    return metrics_dic

es = 2
lr = 4e-5
bs = 16
epochs = 5
frozen_layers = 1
metric_for_best_model = 'pearsonr'
dev = True

tokenize_col='essay'
max_n_tokens=512
extra_cols_for_dataset=['essay_set','norm_scores']

params = generate_parameters(es,lr,bs,epochs,frozen_layers,metric_for_best_model,dev,max_n_tokens)

pre_trained_model_name = 'bert-base-uncased'
dataset_title = 'asap'
task = 'aes'
subtask = 'regression'
abrev_name = f'r2_bert_adaptors_{dataset_title}'

file_paths = generate_file_paths(dataset_title,task,subtask,params,abrev_name)

dataset_dic = {'train':file_paths['train_file_path'],'test':file_paths['test_file_path'],'val':file_paths['val_file_path']}

dataset = load_dataset('csv', data_files=dataset_dic)
tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
dataset = dataset.map(encode_batch, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask","labels"]+extra_cols_for_dataset)

min_max_dic = read_pickle_file(f'{file_paths["pickle_files"]}nomalised_params.pickle')

adaptor_name = f'{this_model}_{dataset_title}'
model = R2BERTAdaptors(model_name,norm_params=min_max_dic,dynamic=True,adaptor_names=adaptor_name)
model.activate_adapters()
model = model.model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

metric = load_metric("spearmanr")

training_args = base_training_args(params,file_paths)

callbacks = [EarlyStoppingCallback(es)]

trainer = R2Trainer(model=model,
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