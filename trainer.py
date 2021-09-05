import os
os.chdir('/content/drive/MyDrive/CAMemBERT2')
from transformers import TrainingArguments,Trainer,AutoModelForSequenceClassification,AutoModel
import numpy as np
from math import floor
import torch
from torch.cuda import is_available
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score,classification_report,fbeta_score
import time 
import json
import pdb

class MultiTaskModelTrainer(Trainer):

    prev_epoch = 0
    device = 'cuda' if is_available() else 'cpu'
    normalise_values = None

    def compute_loss(self,model,inputs,return_outputs=False):
        if self.state.global_step==0:
            self.start_time = time.time()
    	# round down epoch 
        epoch = floor(self.state.epoch)
        # calculate the current step within an epoch
        step_in_epoch = self.state.global_step+1-(epoch*(self.state.max_steps/self.state.num_train_epochs))
        if step_in_epoch==1:
            print('1st step')
            self.model.train()   
            # inits for start of training
            if self.state.global_step == 0:
                # avg cost across predetermined number of epochs (init as zero for first epoch) used for dwa
                self.avg_cost = torch.zeros([self.state.num_train_epochs, len(self.kwargs_['tasks'])]).float().to(self.device)
                # a tensoer where each col represetnts the task and each row represents its value at a given epoch
                indexes , weights = zip(*sorted([(self.kwargs_['map_task_index'][task],self.kwargs_['init_task_weightings'][task]) for task in self.kwargs_['tasks']]))
                self.lambda_weights = torch.tensor(list(weights)).expand(self.state.num_train_epochs,-1).to(self.device)

        if self.kwargs_['optimizer_weighting']=='dwa':
            # if the first step of any epoch update task weighing values 
            if step_in_epoch==1:
                if epoch not in list(range(self.kwargs_['epochs_to_avg_over'])):
                    # find the change in loss over the past number of spesified epochs for each task
                    ws = [torch.exp((self.avg_cost[epoch - 1, val] / self.avg_cost[epoch - self.kwargs_['epochs_to_avg_over'], val])/self.kwargs_['temp']) for val in range(len(self.kwargs_['tasks']))]
                    # updates weights in accordance with change in loss
                    self.lambda_weights[epoch, :] = torch.tensor([len(self.kwargs_['tasks']) * (ws[val]/sum(ws)) for val in range(len(self.kwargs_['tasks']))])

        if self.kwargs_['optimizer_weighting']=='dynamic':
            if step_in_epoch==1:
                self.update_te(epoch)
                self.lambda_weights[epoch, :] = torch.tensor([self.calc_dynamic_loss(task,epoch) for task in self.kwargs_['tasks']])


        outputs = self.model(**inputs)
        # get training loss for current task based off of current mini batch
        loss =  self.sum_losses(outputs,epoch)

        # if final step in epoch perform evaluation
        if (self.state.global_step+1)%(self.state.max_steps/self.state.num_train_epochs)==0 or self.state.global_step == self.state.max_steps-1:
            self.compute_metrics_2(eval_dataset=self.eval_dataset,epoch=floor(self.state.epoch),testing=False)
        return loss

    def sum_losses(self,outputs,epoch,eval=False):
        train_loss = {}
        for task in self.kwargs_['tasks']:
            try:
                output_loss = outputs[f'{task}_loss']
                train_loss[task] = output_loss
            except:
                continue
        
        losses = {}
        for k,v in train_loss.items():
            losses[k] = self.lambda_weights[epoch, self.kwargs_['map_task_index'][k]] * v
            if self.kwargs_['optimizer_weighting']=='dwa':
                self.avg_cost[epoch, self.kwargs_['map_task_index'][k]] += train_loss[k].item() / (self.state.max_steps/self.state.num_train_epochs)
        if eval:
            return losses
        else:
            return sum(losses.values())


    def compute_metrics_2(self,eval_dataset,epoch,testing=False):
        
        self.model.eval()
        history = {task:defaultdict(list) for task in self.kwargs_['tasks']}
        losses = defaultdict(int)
        total_loss = 0.0
        # add all outputs to device
        for step,inputs in enumerate(self.get_eval_dataloader(eval_dataset)):     
            for key, value in inputs.items():
                inputs[key] = inputs[key].to(self.device)
            # get output from model using mini batch
            outputs = self.model(**inputs)
            calculated_loses = self.sum_losses(outputs,epoch,eval=True)

            for task,value in calculated_loses.items():
                # add loss to the total loss of the model and inidividual task losses
                losses[task] += value.item()
                total_loss += value.item()
            for key,value in outputs.items():
                # appnd the list of predictions based off the task and output
                if 'loss' not in key and 'dataset' != key:
                    history[self.kwargs_['map_out_to_task'][key]][self.kwargs_['map_out_to_out'][key]] = history[self.kwargs_['map_out_to_task'][key]][self.kwargs_['map_out_to_out'][key]] + value.cpu().detach().numpy().tolist()

        weights_dic = {}
        for task in self.kwargs_['tasks']:
            weights_dic[f'{task}_weight_coef'] = self.lambda_weights[epoch,self.kwargs_["map_task_index"][task] ].item()
            if testing== False:
                print( f'{task}_weight_coef : {self.lambda_weights[epoch,self.kwargs_["map_task_index"][task] ]}')
        print()
        avg_losses = {k:v/step for k,v in losses.items()}
        avg_losses['total_loss'] = total_loss/step
        if testing== False:
            print('losses',avg_losses)
            print()

        metrics_dics = []
        for task,values in history.items():
            if task in self.kwargs_['regression_tasks']:
                if self.normalise_values:
                    metrics_dics.append(self.calc_regression_metrics(values['preds'],values['labels'],task,testing))
                else:
                    metrics_dics.append(self.calc_regression_metrics(np.rint(values['preds']),values['labels'],task,testing))
            elif task in self.kwargs_['classification_tasks']:
                metrics_dics.append(self.calc_classification_metrics(values['preds'],values['labels'],task,testing))
        
            logs = {k:v for dic in metrics_dics for k,v in dic.items()}
            logs = {**logs,**avg_losses,**weights_dic}
        if testing==False:
            self.log(logs)
            log_hist = [epoc for epoc in self.state.log_history if 'learning_rate' not in epoc.keys()]
            self.best_metrics_values = {task : max([epoc[self.kwargs_['metrics_to_track_by_task'][task]] for epoc in log_hist]) for task in self.kwargs_['tasks'] }
            self.best_metrics_epoch = {task : np.argmax([epoc[self.kwargs_['metrics_to_track_by_task'][task]] for epoc in log_hist]) for task in self.kwargs_['tasks'] }
            if len(self.kwargs_[f'tasks'])>1:
                self.lowest_loss_epoch = np.argmin([epoc['total_loss'] for epoc in log_hist])
            else:
                self.lowest_loss_epoch = np.argmin([epoc[f'{self.kwargs_["task_priorities_priority_as_key"]["primary_task"]}'] for epoc in log_hist])

            if self.kwargs_['early_stopping_patience']:
                train_hist = [epoc for epoc in self.state.log_history if 'learning_rate' in epoc.keys()]
                if self.kwargs_['early_stopping_metric'] == 'loss':
                    print('epochs since lowest loss',(floor(self.state.epoch) - self.lowest_loss_epoch))
                    print()
                    if (floor(self.state.epoch) - self.lowest_loss_epoch) >= self.kwargs_['early_stopping_patience']:
                        self.save_hist_and_stop_training(log_hist,train_hist)

                else:
                    print('epochs since best performance',(floor(self.state.epoch) - self.best_metrics_epoch[self.kwargs_['task_priorities_priority_as_key']['primary_task']]))
                    print()
                    if (floor(self.state.epoch) - self.best_metrics_epoch[self.kwargs_['task_priorities_priority_as_key']['primary_task']]) >= self.kwargs_['early_stopping_patience']:
                        self.save_hist_and_stop_training(log_hist,train_hist)
            if self.state.global_step == self.state.max_steps-1:
                train_hist = [epoc for epoc in self.state.log_history if 'learning_rate' in epoc.keys()]
                self.save_hist_and_stop_training(log_hist,train_hist)
        else:
            train_hist = [epoc for epoc in self.state.log_history if 'learning_rate' in epoc.keys()]
            return logs,{task:{'preds':values['preds'],'labels':values['labels']} for task,values in history.items() if task in self.kwargs_['regression_tasks']}
    
    def save_hist_and_stop_training(self,log_hist,train_log):
        best_hist = {task : {metric:value for metric,value in log_hist[epoch].items() if task in metric} for task,epoch in self.best_metrics_epoch.items()}
        train_hist = self.convert_to_list_dict(train_log)
        eval_hist = self.convert_to_list_dict(log_hist)

        test_hist,test_preds = self.compute_metrics_2(eval_dataset=self.test_dataset,epoch=self.best_metrics_epoch[self.kwargs_["task_priorities_priority_as_key"]["primary_task"]],testing=True)
        info_dict = self.kwargs_
        info_dict['learning_rate'] = self.args.learning_rate
        info_dict['batch_size'] = self.args.per_device_train_batch_size
        info_dict['weight_decay'] = self.args.weight_decay
        info_dict['adam_epsilon'] = self.args.adam_epsilon
        info_dict['normalized_values'] = self.normalise_values
        time_stamp = str(time.time()).split('.')[0]
        info_dict['model_name'] = f'{self.kwargs_["pretrained_model"]}_{time_stamp}'
        info_dict['runtime'] = (time.time() - self.start_time)
        info_dict['steps_per_sec'] = info_dict['runtime']/self.state.global_step
        hist = {}
        hist['best'],hist['train'],hist['eval'],hist['test'],hist['info'],hist['preds'] = best_hist,train_hist,eval_hist,test_hist,info_dict,test_preds
        print('test results',hist['test'])
        self.history_dict = hist
        with open(f'results/raw_results/{info_dict["model_name"]}.json', 'w') as fp:
            json.dump(hist, fp)
            print('model saved to results/raw_results/',time_stamp)
        self.state.max_steps = self.state.global_step 
        
    def convert_to_list_dict(self,log):
        hist = defaultdict(list)
        [hist[k].append(v) for epoch in log for k,v in epoch.items()]
        return hist

    def calc_regression_metrics(self,preds,labels,task,testing):
        if self.normalise_values:
            preds,labels = [np.rint((self.normalise_values[0]*np.array(nums))+self.normalise_values[1]) for nums in [preds,labels]]

        metrics_dic = {
            f"rmse_{task}": np.sqrt(np.mean((preds-labels)**2)),
            f"pearson_{task}": np.corrcoef(preds,labels)[0,1],
            f"spearman_{task}" : spearmanr(preds, labels)[0],
            f"kappa_{task}":cohen_kappa_score(preds,labels,weights='quadratic')
            }
        if testing== False:
            print(f'{task}_metrics',metrics_dic)
            print()
        return metrics_dic

    def calc_classification_metrics(self,preds,labels,task,testing):
        digits = 9 if task == 'ner' else 2
        if testing== False:
            print(task,classification_report(labels, preds, digits=digits))
        report_output = classification_report(labels, preds, digits=digits, output_dict=True)
        metrics_dic = {
          f'f1_score_avg_{task}' : report_output['accuracy'],
          f'f1_score_macro_{task}' : report_output['macro avg']['f1-score'],
          f'f1_score_weighted_{task}' : report_output['weighted avg']['f1-score'],
    	  }
        if digits==2:
            metrics_dic[f'f_0_5_{task}'] = fbeta_score(preds,labels,beta=0.5)
        if testing== False:
            print(f'{task}_metrics',metrics_dic )
            print()
        return metrics_dic

    def calc_dynamic_loss(self,task,epoch):
        if task in self.kwargs_['regression_tasks']:
            return self.te
        elif task in self.kwargs_['classification_tasks']:
            if self.kwargs_['task_priorities_priority_as_key']['secondary_task']==task:
                return (1-self.te)
            else:
                return (1-self.te)*0.1

    def update_te(self,epoch):
        if epoch==0:
            self.gamma = np.log((1/1e-6)-1)/((self.args.num_train_epochs/2)-1)
        self.te = 1/(1+np.exp(self.gamma*((self.args.num_train_epochs/2)-epoch)))

    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self,eval_dataset):
        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )