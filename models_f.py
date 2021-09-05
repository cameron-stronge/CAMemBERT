# import os
# os.chdir('/content/drive/MyDrive/CAMemBERT2')
import pandas as pd
import numpy as np
import pdb
import re
import string
import copy
from transformers import AutoModelForSequenceClassification,TrainingArguments,Trainer,AutoTokenizer,AutoModelForTokenClassification,AutoModel,AutoConfig,EarlyStoppingCallback
from transformers.models.bert.modeling_bert import TokenClassifierOutput
from datasets import DatasetDict,Dataset,load_dataset
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from math import floor
import torch 
from torch import nn


class ClassificationHead(nn.Module):

    # initialise classification head (only if 'aes' is not the primary task )
    def __init__(self,task_name,mini_task_dict,pretrained_model_name='distilroberta-base',shared_encoder_layer=None):
        super().__init__()
        # loads model from pretrained huggingface model (for aes a classification head with one label is expected)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = mini_task_dict['model'].from_pretrained(pretrained_model_name,
                                                        num_labels=mini_task_dict['n_labels'],
                                                        output_hidden_states=True)
        # extracts drop out layer amd classifier from model, l(there are definitely less computationally expensive ways to get a classifier, but this tended to yield best results from past experience)
        self.dropout = nn.Dropout(model.config.hidden_dropout_prob)
        self.classifier = model.classifier.to(self.device)
        # defines loss function as mean square error loss
        self.loss_fct = nn.MSELoss()

    def forward(self,original_model_output,inputs):
        # pass the output of the layer where classification head is positioned through the dropout and linear layer to get a predicted score for each input in the minibatch
        # compare predictions to labels to calculate the loss.
        sequence_output = self.dropout(original_model_output)
        logits = self.classifier(sequence_output)
        loss = loss_fct(logits,inputs['labels'])
        return {'loss':loss,'preds':logits,'labels':inputs['labels']}

class TaggingHead(nn.Module):

    # create a classifier using that predicts the class that each token belongs to
    # i.e for a sequnce of 512 tokens that could belong to 2 classes the linear layer calculates a probability of every token belonging to class 1 or 2 (would end up 1024 probabilities)
    # the cross entropy loss uses all these probabilities to calculate the loss.
    def __init__(self,task_name,mini_task_dict,pretrained_model_name='distilroberta-base',shared_encoder_layer=None):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = AutoConfig.from_pretrained(pretrained_model_name,
                                                        num_labels=mini_task_dict['n_labels'],
                                                        output_hidden_states=True)
        self.config = config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels).to(self.device)
        self.class_weights = mini_task_dict['class_weights']
        # apply class weightings to the loss function to heavily penalise the misclassification of underrepresented classes
        self.loss_fct = nn.CrossEntropyLoss(weight = self.class_weights) if self.class_weights != None else nn.CrossEntropyLoss()
        self.shared_encoder_layer = shared_encoder_layer

    # the tagging head might be part of a shared encoder layer which might be an extension to the original model.
    def set_shared_encoder_layer(self,encoder_layer):
        self.shared_encoder_layer = encoder_layer

    def forward(self,original_model_output,inputs):
        output = self.shared_encoder_layer(original_model_output).last_hidden_state if self.shared_encoder_layer else original_model_output
        sequence_output = self.dropout(output)
        logits = self.classifier(sequence_output)
        loss = None
        labels = inputs['labels']
        # Only keep active parts of the loss (non 0's in attention mask)
        active_loss = inputs['attention_mask'].view(-1) == 1
        # flatten logits to be in one column of prediction pairs
        active_logits = logits.view(-1, self.config.num_labels)
        # reshape loss to only consist of active labels
        active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(self.loss_fct.ignore_index).type_as(labels))
        # apply a mask to remove padded token added as part of pre processing
        non_padded_mask = active_labels!=-100
        loss = self.loss_fct(active_logits[non_padded_mask], active_labels[non_padded_mask])
        # create a mask to apply to predictions to keep only active non padded tokens 
        active_preds_mask = torch.logical_and(active_loss,non_padded_mask)
        # predictions made by finding the argmax for each probability for each token i.e selecting the max between [0.3,0.7] would result in a prediction of class 2 instead of class 1
        return {'loss':loss,'preds':torch.argmax(logits,2).flatten()[active_preds_mask],'labels':active_labels[non_padded_mask]}

class MultiTaskModel(nn.Module):

    def __init__(self,pretrained_model='distilroberta-base',kwargs_dict=None,current_task=None):
        super().__init__()
        self.kwargs_dict = kwargs_dict
        self.primary_task = self.kwargs_dict['task_priorities_priority_as_key']['primary_task']
        # uses primary task model as main model
        self.model = self.kwargs_dict[self.primary_task]['model'].from_pretrained(pretrained_model,
                                                               num_labels=self.kwargs_dict[self.primary_task]['n_labels'],
                                                               output_hidden_states=True)
        self.shared_encoder_layer = None
        self.current_task = current_task
        self.decoder_dict = {}
        # if the model is to contain a shared encoder (meaning an additional encoder layer(s) that are used by two or more tasks)
        # the encoder is initialised here and then used by multiple heads by passing the hidden output of the bert model through this layer before calculating the loss at the head
        # the layer can be moved so it could use the hidden output of the model at the first layer or the third as an example.
        if self.kwargs_dict['shared_encoder_n_layers']>0 and self.shared_encoder_layer==None:
            # input paramteres of shared encoder (how many layers should it consist of)
            config = AutoConfig.from_pretrained(pretrained_model,num_hidden_layers=self.kwargs_dict['shared_encoder_n_layers'],output_hidden_states=True)
            shared_encoder = AutoModel.from_pretrained(pretrained_model,config=config).encoder
            self.shared_encoder_layer = shared_encoder
        
        # creating a tagging or classification head for each task 
        for task in self.kwargs_dict['tasks']:
            if task in self.kwargs_dict['classification_tasks']:
                self.decoder_dict[task] = TaggingHead(task,self.kwargs_dict[task])
                if self.kwargs_dict[task]['shares_encoder'] and self.shared_encoder_layer:
                    self.decoder_dict[task].set_shared_encoder_layer(self.shared_encoder_layer)
            elif task in self.kwargs_dict['regression_tasks'] and task != self.primary_task :
                self.decoder_dict[task] = ClassificationHead(task_name,self.kwargs_dict[task])

    # where the inputs of the min batch are padded to the model.
    def forward(self,**inputs):
        # check to see if the mini batch comes from one datase
        dataset = torch.unique(inputs['dataset'])[0] if len(torch.unique(inputs['dataset']))==1 else 'mix'
        # if there is just one dataset in a minibatch get the preds labels and loss calculated by the model.
        if dataset!='mix':
            outputs_dict = self.get_outputs(inputs,dataset)
        else:
            outputs_dict = {}
             # if there is more then one dataset in a minibatch (possible at the end of training) then split the inputs and feed the through the model as two seperate mini batches
            for task in self.split_inputs(inputs):
                outputs = self.get_outputs(inputs=task[1],dataset=0) if task[0]=='fce_task' else self.get_outputs(inputs=task[1],dataset=1) 
                outputs_dict = {**outputs_dict,**outputs}
        return outputs_dict


    def get_outputs(self,inputs,dataset):
        if dataset==0:
            if self.primary_task in self.kwargs_dict['regression_tasks']:
                # ;use autoclassification mode tho get outputs (labels,loss,pred) if primary task is aes

                model_output = self.model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],labels=inputs[self.kwargs_dict[self.primary_task]['labels']].float())
                outputs_dict = {
                                f'{self.primary_task}_loss':model_output.loss,
                                f'{self.primary_task}_preds':model_output.logits.flatten(),
                                f'{self.primary_task}_labels':inputs[self.kwargs_dict[self.primary_task]['labels']]
                                }
            else:
                #else get the model output of the last hidden state
                model_output = self.model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
                outputs_dict = {}

            # if aes is not part of the model extract labels,loss,pred from model output by passing it through the shared encoder if it is present and then through the task head
            tasks = set(self.kwargs_dict['tasks'])-set(['ner'])-set([self.kwargs_dict['task_priorities_priority_as_key']['primary_task']]) if self.kwargs_dict['task_priorities_priority_as_key']['primary_task'] == 'aes' else set(self.kwargs_dict['tasks'])-set(['ner'])
            for task in tasks:
                output = self.decoder_dict[task](model_output.hidden_states[self.kwargs_dict[task]['output_layer']],inputs)
                outputs_dict = self.get_output_dict(task,output,outputs_dict)
        else:
            # if the data is part of the ner dataset extract labels,loss,pred from model output by passing it through the shared encoder if it is present and then through the task head
            if 'ner' in self.kwargs_dict['tasks']:
                model_output = self.model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
                task='ner'
                output = self.decoder_dict[task](model_output.hidden_states[self.kwargs_dict[task]['output_layer']],inputs)
                outputs_dict = self.get_output_dict(task,output,outputs_dict={})
            else:
                raise KeyError
        return outputs_dict

    def get_output_dict(self,task,output,outputs_dict):
        return {**outputs_dict,**{f'{task}_loss':output['loss'],f'{task}_preds':output['preds'],f'{task}_labels':output[self.kwargs_dict[task]['labels']]}}

    # split input for mixed ini-batches
    def split_inputs(self,inputs):
        split_mask = inputs['dataset']==0
        return ((['fce_task'],{k:v[split_mask] for k,v in inputs.items()}),('ner',{k:v[~split_mask] for k,v in inputs.items()}))