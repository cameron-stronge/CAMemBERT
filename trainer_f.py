# import os
# os.chdir('/content/drive/MyDrive/CAMemBERT2')
# imports
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


# This trainer inherits from the trainer from the huggingface library. 
# This class adds dditional functionality to the trainer from the huggingface library so that it works in conjunction with multi-task objectives. 
class MultiTaskModelTrainer(Trainer):

    # class variables
    prev_epoch = 0
    device = 'cuda' if is_available() else 'cpu'
    # this variable is desigened to be set outside the model
    # not at all the most pythonic way to use anf initialise this dictionary, but was a necessary evil to work around
    # the difiulty of initialising the trainer class.
    kwargs_dict = None
    
    # method is called for each mini-batch that is to be passed through the model
    def compute_loss(self,model,inputs,return_outputs=False):
    	
        # round down epoch of current step (otherwise is a float value)
        epoch = floor(self.state.epoch)
        # calculate the current step within an epoch
        step_in_epoch = self.state.global_step+1-(epoch*(self.state.max_steps/self.state.num_train_epochs))
        if step_in_epoch==1:
            # if first step in epoch make sure model is set to train plus initialise required variables
            self.model.train()   
            # inits for start of training
            if self.state.global_step == 0:
                # avg cost across predetermined number of epochs (init as zero for first epoch) used for dwa
                self.avg_loss = torch.zeros([self.state.num_train_epochs, len(self.kwargs_dict['tasks'])]).float().to(self.device)
                # for weights a col represetnts a weight to multiply a task loss by and each row represents that value at a given epoch
                # so for example a model is initiated with weights [1,0.5,0.2] for tasks ['aes','ged','ner'] and is to be trained for 2 epochs:
                # [[1,0.5,0.2],
                #  [1,0.5,0.2]]
                # Useful for dynamic average weightings so that epoch weights can be updated and accessed at start of each epoch 
                # 
                # to ensure that the weightints are at the correct index their position is determined by the order of tasks defined in the trainer arguments
                indexes , weights = zip(*sorted([(self.kwargs_dict['map_task_index'][task],self.kwargs_dict['init_task_weightings'][task]) for task in self.kwargs_dict['tasks']]))
                self.lambda_weights = torch.tensor(list(weights)).expand(self.state.num_train_epochs,-1).to(self.device)
                # start time for runtime
                self.start_time = time.time()

        if self.kwargs_dict['optimizer_weighting']=='dwa':
            # if the first step of any epoch update task weighing values 
            if step_in_epoch==1:
                if epoch not in list(range(self.kwargs_dict['epochs_to_avg_over'])):
                    # find the change in loss over the past number of spesified epochs for each task
                    ws = [torch.exp((self.avg_loss[epoch - 1, val] / self.avg_loss[epoch - self.kwargs_dict['epochs_to_avg_over'], val])/self.kwargs_dict['temp']) for val in range(len(self.kwargs_dict['tasks']))]
                    # updates weights in accordance with change in loss
                    self.lambda_weights[epoch, :] = torch.tensor([len(self.kwargs_dict['tasks']) * (ws[val]/sum(ws)) for val in range(len(self.kwargs_dict['tasks']))])

        # if the weights are to change during training (by increasing weigting of loss from less to more complex task)
        if self.kwargs_dict['optimizer_weighting']=='dynamic':
            if step_in_epoch==1:
                # if at the beggining of the epoch update factor for weighting loss of less and more complex task
                self.update_te(epoch)
                # update tensor that determines weights accordingly
                self.lambda_weights[epoch, :] = torch.tensor([self.calc_dynamic_loss(task,epoch) for task in self.kwargs_dict['tasks']])

        # get model output
        outputs = self.model(**inputs)
        # get training loss for current task based off of current mini batch
        loss =  self.sum_losses(outputs,epoch)

        # if final step in epoch perform evaluation : 
        if (self.state.global_step+1)%(self.state.max_steps/self.state.num_train_epochs)==0 or self.state.global_step == self.state.max_steps-1:
            # the original trainer from huggingface struggled with the requirements of evaluating metrics such as the potential for mixed dataset inputs
            # (i.e a mini batch containing 4 samples from fce dataset and 4 from conll2003 dataset) so to cope a new evaluation method was added
            self.compute_metrics_2(eval_dataset=self.eval_dataset,epoch=floor(self.state.epoch),testing=False)
        return loss

    def sum_losses(self,outputs,epoch,eval=False):
        train_loss = {}
        # iterate over task
        for task in self.kwargs_dict['tasks']:
            # see if the current mini-batch returns a loss associated with task in iteration
            try:
                # store the loss for the current mini-batch
                train_loss[task] = outputs[f'{task}_loss']
            except:
                continue
        
        losses = {}
        # iterate through each loss from the mini-batch
        for task,loss_value in train_loss.items():
            # multiply the loss value by the weighting for the task at the current epoch
            losses[task] = self.lambda_weights[epoch, self.kwargs_dict['map_task_index'][task]] * loss_value
            # if optimizer weighting is dynamic weighted average update the loss for each task to keep track of the average loss (for each task) across the whole epoch 
            if self.kwargs_dict['optimizer_weighting']=='dwa':
                self.avg_loss[epoch, self.kwargs_dict['map_task_index'][task]] += train_loss[task].item() / (self.state.max_steps/self.state.num_train_epochs)
        # if evaluating return losses as a dict else return the sum of all the losses per task
        return losses if eval else sum(losses.values())

    def compute_metrics_2(self,eval_dataset,epoch,testing=False):
        
        # set model to evaluation (no weight updates)
        self.model.eval()
        # history: stores a list of all the prediction and labels for each sample in the dataset to be evaluated for each task
        history = {task:defaultdict(list) for task in self.kwargs_dict['tasks']}
        # create a dictionary of losses for each task
        losses = defaultdict(int)
        total_loss = 0.0
        # add all inputs to device and iterate through each mini in the current eval or test dataset
        for step,inputs in enumerate(self.get_eval_dataloader(eval_dataset)):     
            for key, value in inputs.items():
                inputs[key] = inputs[key].to(self.device)
            # get output from model using mini batch
            outputs = self.model(**inputs)
            # calculate losses in eval mode
            calculated_loses = self.sum_losses(outputs,epoch,eval=True)

            # ge the total loss for each task
            for task,value in calculated_loses.items():
                # add loss to the total loss of the model and inidividual task losses
                losses[task] += value.item()
                # add the loss for each individula task to the total loss
                total_loss += value.item()
            #iterate each output which could be _loss, _preds or _labels for each
            for key,value in outputs.items():
                # appnd the output to the correct position in th history dictionary (but not losses)
                if 'loss' not in key and 'dataset' != key:
                    # an example of how this works:
                    # key = 'aes_preds'
                    # value = [24,12,...] (score predictions)
                    # self.kwargs_dict['map_out_to_task'][aes_preds] => 'aes'
                    # self.kwargs_dict['map_out_to_out'][aes_preds] => 'preds'
                    # therefore history['aes']['preds'] = history['aes']['preds'] (which would be a numpy array of all previous aes predictions seen) + the curent predictions
                    history[self.kwargs_dict['map_out_to_task'][key]][self.kwargs_dict['map_out_to_out'][key]] = history[self.kwargs_dict['map_out_to_task'][key]][self.kwargs_dict['map_out_to_out'][key]] + value.cpu().detach().numpy().tolist()

        weights_dic = {}
        # iterate over each task and get its current weight coefficient for this epoch
        for task in self.kwargs_dict['tasks']:
            weights_dic[f'{task}_weight_coef'] = self.lambda_weights[epoch,self.kwargs_dict["map_task_index"][task] ].item()
            if testing== False:
                print( f'{task}_weight_coef : {self.lambda_weights[epoch,self.kwargs_dict["map_task_index"][task] ]}')
        print()
        # average the loss for each task by dividing by the number of mini batches in the eval / test dataset
        avg_losses = {k:v/step for k,v in losses.items()}
        avg_losses['total_loss'] = total_loss/step
        if testing== False:
            print('losses',avg_losses)
            print()

        # calculate the performance metrics for each task and store results in a list
        metrics_dics = []
        for task,values in history.items():
            # return appropriate metrics based on whether task is regression or classification
            if task in self.kwargs_dict['regression_tasks']:
                # if there are normalisation values then scores are denormalised
                if self.kwargs_dict['normalised_values']:
                    metrics_dics.append(self.calc_regression_metrics(values['preds'],values['labels'],task,testing))
                else:
                    metrics_dics.append(self.calc_regression_metrics(np.rint(values['preds']),values['labels'],task,testing))
            elif task in self.kwargs_dict['classification_tasks']:
                metrics_dics.append(self.calc_classification_metrics(values['preds'],values['labels'],task,testing))
        
            # create log each metric , weight and loss 
            logs = {k:v for dic in metrics_dics for k,v in dic.items()}
            logs = {**logs,**avg_losses,**weights_dic}

        if testing==False:
            # add log dict to self.state.log_history
            self.log(logs)
            # get all the records (dictionaries) in the self.state.log_history that do not contain learning rate as these are the training evaluation data logged automaticall by the trainer
            log_hist = [epoc for epoc in self.state.log_history if 'learning_rate' not in epoc.keys()]
            # find the maximum value for each metric recorded in the self.state.log_history for each task such as max f1_score for ged 
            self.best_metrics_values = {task : max([epoc[self.kwargs_dict['metrics_to_track_by_task'][task]] for epoc in log_hist]) for task in self.kwargs_dict['tasks'] }
            # find the epoch where that maximum value occured
            self.best_metrics_epoch = {task : np.argmax([epoc[self.kwargs_dict['metrics_to_track_by_task'][task]] for epoc in log_hist]) for task in self.kwargs_dict['tasks'] }
            # if there are more then one task find the epoch lowest total_loss else use the loss of the individual task (to be used for early stopping)
            if len(self.kwargs_dict[f'tasks'])>1:
                self.lowest_loss_epoch = np.argmin([epoc['total_loss'] for epoc in log_hist])
            else:
                self.lowest_loss_epoch = np.argmin([epoc[f'{self.kwargs_dict["task_priorities_priority_as_key"]["primary_task"]}'] for epoc in log_hist])

            # if early stopping patience has been provided
            if self.kwargs_dict['early_stopping_patience']:
                # if the metric to track is loss then find the number of epochs that have passed since the lowest loss occured and if > early stopping then stop training and log test metrics
                if self.kwargs_dict['early_stopping_metric'] == 'loss':
                    print('epochs since lowest loss',(floor(self.state.epoch) - self.lowest_loss_epoch))
                    print()
                    if (floor(self.state.epoch) - self.lowest_loss_epoch) >= self.kwargs_dict['early_stopping_patience']:
                        # get the training history
                        train_hist = [epoc for epoc in self.state.log_history if 'learning_rate' in epoc.keys()]
                        self.save_hist_and_stop_training(log_hist,train_hist)
                # if the no metric to track is provided then find the number of epochs that have passed since the best performance of the primary task have occured and if > early stopping then stop training and log test metrics
                else:
                    print('epochs since best performance',(floor(self.state.epoch) - self.best_metrics_epoch[self.kwargs_dict['task_priorities_priority_as_key']['primary_task']]))
                    print()
                    if (floor(self.state.epoch) - self.best_metrics_epoch[self.kwargs_dict['task_priorities_priority_as_key']['primary_task']]) >= self.kwargs_dict['early_stopping_patience']:
                        # get the training history
                        train_hist = [epoc for epoc in self.state.log_history if 'learning_rate' in epoc.keys()]
                        self.save_hist_and_stop_training(log_hist,train_hist)
            # if last epoch of training and penultimate step then stop training and log test metrics
            if self.state.global_step == self.state.max_steps-1:
                # get the training history
                train_hist = [epoc for epoc in self.state.log_history if 'learning_rate' in epoc.keys()]
                self.save_hist_and_stop_training(log_hist,train_hist)
        else:
            # if testing then return the logged data for test data.
            return logs,{task:{'preds':values['preds'],'labels':values['labels']} for task,values in history.items() if task in self.kwargs_dict['regression_tasks']}
    
    def save_hist_and_stop_training(self,log_hist,train_log):
        # find the best value for each metric
        best_hist = {task : {metric:value for metric,value in log_hist[epoch].items() if task in metric} for task,epoch in self.best_metrics_epoch.items()}
        # convert logs which are list of dictionaries to a dictionary of lists
        train_hist = self.convert_to_list_dict(train_log)
        eval_hist = self.convert_to_list_dict(log_hist)
        # get metrics for test data
        test_hist,test_preds = self.compute_metrics_2(eval_dataset=self.test_dataset,epoch=self.best_metrics_epoch[self.kwargs_dict["task_priorities_priority_as_key"]["primary_task"]],testing=True)
        # store info in regards to training
        info_dict = self.kwargs_dict
        info_dict['learning_rate'],info_dict['batch_size'],info_dict['weight_decay'],info_dict['adam_epsilon'] = self.args.learning_rate,self.args.per_device_train_batch_size,self.args.weight_decay,self.args.adam_epsilon
        time_stamp = str(time.time()).split('.')[0]
        info_dict['runtime'] = (time.time() - self.start_time)
        info_dict['steps_per_sec'] = info_dict['runtime']/self.state.global_step
        # combine all logging /  info dicts into one
        hist = {}
        hist['best'],hist['train'],hist['eval'],hist['test'],hist['info'],hist['preds'] = best_hist,train_hist,eval_hist,test_hist,info_dict,test_preds
        print('test results',hist['test'])
        # save as json file
        self.history_dict = hist

        print(f"""some model info : {info_dict['tasks']}_{info_dict['pretrained_model']}_lr{info_dict['learning_rate']}
        bs{info_dict['batch_size']}_fre{info_dict['frozen_layers']}_share{info_dict['shared_encoder_n_layers']}
        outs{info_dict['output_layer_by_task']}_norm{info_dict['normalised_values']}
        pri{info_dict['task_priorities_priority_as_key']}_opt{info_dict['optimizer_weighting']}
        sco{info_dict['scoring']}_es{info_dict['early_stopping_metric']}""")
        info_dict['model_name'] = time_stamp
        if self.kwargs_dict['save_results']==True:
            with open(f'results/raw_results/{info_dict["model_name"]}.json', 'w') as fp:
                json.dump(hist, fp)
                print('model saved to results/raw_results/',time_stamp)
        # stop training 
        self.state.max_steps = self.state.global_step 
        
    # helper method to reformat list of dictionaries to dictionary of lists
    def convert_to_list_dict(self,log):
        hist = defaultdict(list)
        [hist[k].append(v) for epoch in log for k,v in epoch.items()]
        return hist

    # regression metrics 
    def calc_regression_metrics(self,preds,labels,task,testing):
        if self.kwargs_dict['normalised_values'] and labels[0].is_integer()==False:
            # normalising score by multiplying by the first normalisation value and adding the second
            preds,labels = [np.rint((self.kwargs_dict['normalised_values'] [0]*np.array(nums))+self.kwargs_dict['normalised_values'] [1]) for nums in [preds,labels]]
        else: 
            preds,labels = np.array(preds),np.array(labels)
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

    # classification metrics
    def calc_classification_metrics(self,preds,labels,task,testing):
        # digits = number of labels for each task
        digits = 9 if task == 'ner' else 2
        if testing== False:
            print(task,classification_report(labels, preds, digits=digits))
        report_output = classification_report(labels, preds, digits=digits, output_dict=True)
        metrics_dic = {
          f'accuracy_{task}' : report_output['accuracy'],
          f'f1_score_macro_{task}' : report_output['macro avg']['f1-score'],
          f'f1_score_weighted_{task}' : report_output['weighted avg']['f1-score'],
    	  }
        if digits==2:
            metrics_dic[f'f_0_5_{task}'] = fbeta_score(preds,labels,beta=0.5)
        if testing== False:
            print(f'{task}_metrics',metrics_dic )
            print()
        return metrics_dic

    # provide weightings for dunamic loss
    def calc_dynamic_loss(self,task,epoch):
        if task in self.kwargs_dict['regression_tasks']:
            return self.te
        elif task in self.kwargs_dict['classification_tasks']:
            if self.kwargs_dict['task_priorities_priority_as_key']['aux_task']==task:
                return (1-self.te)*0.1
            else:
                return (1-self.te)

    def update_te(self,epoch):
        if epoch==0:
            self.gamma = np.log((1/1e-6)-1)/((self.args.num_train_epochs/2)-1)
        self.te = 1/(1+np.exp(self.gamma*((self.args.num_train_epochs/2)-epoch)))

    # get dataloader for training
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

    # get dataloader for evaluating
    def get_eval_dataloader(self,eval_dataset):
        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )