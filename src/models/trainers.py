import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
preprocess_mod_ind = currentdir.split('/').index('CAMemBERT')
preprocess_mod_path = '/'.join(currentdir.split('/')[:preprocess_mod_ind+1])
sys.path.append(preprocess_mod_path)
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from transformers import Trainer
import matplotlib.pyplot as plt
import pandas as pd
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataset import Dataset, IterableDataset
import time
import math
import copy
from datasets import load_dataset
from collections import defaultdict

class BaseTrainer(Trainer):

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        # +
        testing = False,
        ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        # + 
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        if testing == False:
            # +
            train_dataloader = self.get_eval_dataloader(self.train_dataset)
            loader_list = [(train_dataloader,'train'),(eval_dataloader,'eval')]
        else:
            loader_list = [(eval_dataloader,'eval')]

        start_time = time.time()

        eval_loop = self.prediction_loop # self.evaluation_loop #self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        # +
        outputs = []
        # ##,+,~
        for loader,metric_key_prefix in loader_list:
            output = eval_loop(
                # ~
                loader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            total_batch_size = self.args.eval_batch_size * self.args.world_size
            
            outputs.append(output)
            # #####

        # +
        if testing==False:
            outputs_metrics = {**outputs[0].metrics,**outputs[1].metrics}
        else:
            outputs_metrics = outputs[0].metrics
        print(outputs_metrics)
        # ~ : output.metrics to outputs_metrics
        self.log(outputs_metrics)

        # if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
        #     # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
        #     xm.master_print(met.metrics_report())

        # ~ : output.metrics to outputs_metrics
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, outputs_metrics)
        # ~ : output.metrics to outputs_metrics
        self._memory_tracker.stop_and_update_metrics(outputs_metrics)
        # ~ : output.metrics to outputs_metrics
        return outputs_metrics

    def set_hist_dic(self):
        history_dict = defaultdict(list)
        for epoch_hist in self.state.log_history[:-1]:
            for train_eval in ['train','eval']:
                for k,v in epoch_hist.items():
                    if train_eval in k and len(history_dict[(k.replace(train_eval,''),train_eval)])<len(self.state.log_history[:-1]):
                        if 'time' not in k or 'second' not in k:
                            history_dict[(k.replace(train_eval,''),train_eval)].append(v)
        print(history_dict)                    
        self.history_dict = history_dict

    def plot_history(self,file_path_for_images,this_model):
        self.set_hist_dic()
        history_dict = self.history_dict.copy()
        for met_train_or_eval in history_dict.keys():
            plt.figure()
            met = met_train_or_eval[0]
            train_or_eval = ['train','eval']
            for train_or_eval_val in ['train','eval']:
                values = self.history_dict[(met,train_or_eval_val)]
                
                plt.plot(range(len(values)),values)
            plt.legend(train_or_eval)
            plt.title(f'epochs vs {met}')
            plt.ylabel(f'{met}')
            plt.xlabel(f'epoch')
            plt.savefig(f'{file_path_for_images}{this_model}_{met}.png')
            plt.show()


class Seq2SeqTrainer(BaseTrainer):

    def compute_loss(self,model,inputs,return_outputs=False):
        bert_output = model(**inputs) 
        if model.class_weights!=None:
            predictions_loss = torch.flatten(bert_output[1],0,1)
            labels=torch.flatten(inputs['labels'])
            loss=F.cross_entropy(predictions_loss,labels,weight=model.class_weights)
            predictions = torch.argmax(bert_output[1],2)
        else:
            loss = bert_output[0]
            predictions = torch.argmax(bert_output[1],2)

        return (loss,(loss,predictions)) if return_outputs else loss

class R2Trainer(BaseTrainer):

    def compute_loss(self,model,inputs,return_outputs=False):
        bert_ouput = model(**inputs)
        batch_size = inputs['input_ids'].size()[0]
        if hasattr(model, 'has_adaptors'):
            bert_ouput = bert_ouput[1]

        if model.norm_params!=None:
            predictions_for_loss = torch.sigmoid(bert_ouput).view(batch_size)
            essay_sets = inputs['essay_set'].data.cpu().numpy()
            predictions_for_loss_cpu = predictions_for_loss.data.cpu().numpy()
            predictions = torch.tensor([self.get_actual_score(model,scores,set_) for scores,set_ in zip(predictions_for_loss_cpu,essay_sets)], dtype=torch.int).cuda().float()
            
            labels_for_loss = inputs['norm_scores'].float()

        else:
            predictions = bert_ouput.view(batch_size)
            predictions_for_loss = bert_ouput.view(batch_size).float()
            
            labels_for_loss = inputs['labels'].float()

        loss_m = F.mse_loss(predictions_for_loss,labels_for_loss)
        # soft max is applied to both predicted and normalised scores (essentially determing the probability
        # that for each essay in the set that it woruld be ranked the highest scoring)
        # This enables the use of the listnet algorithm which is used for ranking loss
        sm_pred_scores = F.softmax(predictions_for_loss,dim=0)
        sm_gold_scores = F.softmax(labels_for_loss,dim=0) 
        # The loss for the listnet function is the cross entropy as applied here, this essentially determines 
        # how different the two soft max distrobutions are
        loss_r = torch.sum((-sm_gold_scores*torch.log(sm_pred_scores)))

        if model.one_loss == 'regression':
            loss = loss_m
            return (loss,(loss,predictions.int())) if return_outputs else loss
        elif model.one_loss == 'ranking':
            loss = loss_r
            return (loss,(loss,predictions.int())) if return_outputs else loss

        if model.dynamic==False:
            loss = loss_m + loss_r
        else:
            loss = self.calc_dynamic_loss(loss_m,loss_r)
        return (loss,(loss,predictions.int())) if return_outputs else loss

    def get_actual_score(self,model,score,essay_set):
        range = model.norm_params['ranges'][essay_set]
        min = model.norm_params['min_scores'][essay_set]
        return np.round(score*range + min)

    def calc_dynamic_loss(self,loss_m,loss_r):
        if self.state.epoch==0:
            self.set_gamma()
            self.update_te()
            self.prev_epoch = 0
        elif self.state.epoch != self.prev_epoch:
            self.prev_epoch = copy.copy(self.state.epoch)
            self.update_te()
        return self.te*loss_m + (1-self.te)*loss_r

    def set_gamma(self):
        self.gamma = np.log((1/1e-6)-1)/((self.args.num_train_epochs/2)-1)

    def update_te(self):
        self.te = 1/(1+np.exp(self.gamma*((self.args.num_train_epochs/2)-self.state.epoch)))