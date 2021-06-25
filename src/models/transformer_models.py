import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
preprocess_mod_ind = currentdir.split('/').index('CAMemBERT')
preprocess_mod_path = '/'.join(currentdir.split('/')[:preprocess_mod_ind+1])
sys.path.append(preprocess_mod_path)
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoConfig,AutoModelForTokenClassification,AutoModel
import re

class TaggingModel(nn.Module):

    def __init__(
        self,
        pretrained_model_name,
        labels,
        class_weights=None
    ):
        # add final layer to make score prediction
        super().__init__()
        self.num_labels = len(labels)
        self.id2label = {id_: label for id_, label in enumerate(labels)}
        self.label2id = {label: id_ for id_, label in enumerate(labels)}
        config = AutoConfig.from_pretrained(pretrained_model_name,num_labels=len(labels), 
                                              id2label={id_: label for id_, label in enumerate(labels)}, 
                                              label2id={label: id_ for id_, label in enumerate(labels)})
        self.config = config
        self.model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name,config=config)
        self.class_weights = class_weights
        

    def set_trainable_params(self,n_training_layer=None):
        for param_name,param_value in self.model.named_parameters():
            if n_training_layer:
                layer_num = re.findall(r'\d+',param_name)
                if len(layer_num)>0:
                    layer_num = int(layer_num[0])
                else:
                    layer_num = 0
                if param_name.startswith('model') and layer_num<n_training_layer:
                    param_value.requires_grad = False
            else:
                if param_name.startswith('model'):
                    param_value.requires_grad = False

    def forward(self,return_dict=True,**inputs):
        bert_output = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['labels'])
        return bert_output

class R2BERT(nn.Module):

    def __init__(
        self,
        pretrained_model_name,
        norm_params = None,
        dynamic = False,
        one_loss = False,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained_model_name)
        self.config = config
        self.model = AutoModel.from_pretrained(pretrained_model_name,config=config)
        self.predictor = nn.Linear(config.hidden_size,1)
        self.norm_params = norm_params
        self.dynamic = dynamic
        self.one_loss = one_loss
    

    def set_trainable_params(self,n_training_layer=None):
        for param_name,param_value in self.model.named_parameters():
            if n_training_layer:
                layer_num = re.findall(r'\d+',param_name)
                if len(layer_num)>0:
                    layer_num = int(layer_num[0])
                else:
                    layer_num = 0
            if param_name.startswith('model') and layer_num<n_training_layer:
                param_value.requires_grad = False
            else:
                if param_name.startswith('model'):
                    param_value.requires_grad = False

    def forward(self,return_dict=True,**inputs):
        bert_output = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'])
        text_representation = bert_output[0][:,0,:]
        return self.predictor(text_representation)