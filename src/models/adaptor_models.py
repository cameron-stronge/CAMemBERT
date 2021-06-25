import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
preprocess_mod_ind = currentdir.split('/').index('CAMemBERT')
preprocess_mod_path = '/'.join(currentdir.split('/')[:preprocess_mod_ind+1])
sys.path.append(preprocess_mod_path)
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoConfig,AutoModelWithHeads

class TaggingModelAdaptors(nn.Module):

    def __init__(
        self,
        pretrained_model_name,
        labels,
        adapter_names,
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
        self.model = AutoModelWithHeads.from_pretrained(pretrained_model_name,config=config)
        self.model.class_weights = class_weights
        self.adapter_names = adapter_names
        self.model.has_adaptors = True
        

    def activate_adapters(self):
        # Add a new adapter
        self.model.add_adapter(self.adapter_names)
        # Add a matching classification head
        self.model.add_tagging_head(
            self.adapter_names,
            num_labels=self.num_labels,
            id2label=self.id2label
        )
        # Activate the adapter
        self.model.train_adapter(self.adapter_names)

class R2BERTAdaptors(nn.Module):

    def __init__(
        self,
        pretrained_model_name,
        norm_params = None,
        dynamic = False,
        one_loss = False,
        adaptor_names = None,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained_model_name)
        self.config = config
        self.model = AutoModelWithHeads.from_pretrained(pretrained_model_name,config=config)
        self.model.norm_params = norm_params
        self.model.dynamic = dynamic
        self.model.one_loss = one_loss
        self.adaptor_names = adaptor_names
        self.model.has_adaptors = True
    
    def activate_adapters(self):
        # Add a new adapter
        self.model.add_adapter(self.adaptor_names)
        # Add a matching classification head
        self.model.add_classification_head(
            self.adaptor_names,
            num_labels=1
        )
        # Activate the adapter
        self.model.train_adapter(self.adaptor_names)
