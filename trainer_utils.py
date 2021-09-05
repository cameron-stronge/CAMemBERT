from transformers import AutoModel,AutoModelForSequenceClassification,TrainingArguments
from preprocessing_f import CreateHuggingFaceMultiTask
import pdb
from sklearn.model_selection import ParameterGrid

# training arguments for model
def training_args(batch_size=8,save_strategy='no',output_dir='/',lr= 5e-4,epochs=30,weight_decay=0.01):
	args = TrainingArguments(
	    output_dir=output_dir,
	    save_strategy=save_strategy,
	    evaluation_strategy='no',
	    learning_rate=lr,
	    per_device_train_batch_size=batch_size,
	    per_device_eval_batch_size=batch_size,
	    num_train_epochs=epochs,
	    weight_decay=weight_decay,
	)
	return args

# trainin kwargs for trainer
def training_kwargs(
    tasks=['aes','ged','ner'],outputs=['loss','labels','preds'],optimizer_weighting='fixed',
    init_task_weightings={'aes':1.0,'ged':1.0,'ner':1.0},task_priorities={'aes':'primary_task','ged':'secondary_task','ner':'aux_task'},
    temp = 2,metrics_to_track_by_task={'aes':'pearson_aes','ged':'f_0_5_ged','ner':'f1_score_macro_ner'},
    regression_tasks=['aes'],classification_tasks=['ner','ged'],fce_tasks=['aes','ged'], 
    class_weights={'aes':None,'ged':None,'ner':None},n_labels={'aes':1,'ged':2,'ner':9},labels={'aes':'scores','ged':'labels','ner':'labels'},
    model={'aes':AutoModelForSequenceClassification,'ged':AutoModel,'ner':AutoModel},
    shares_encoder={'aes':False,'ged':True,'ner':True},output_layer={'aes':-1,'ged':-1,'ner':-1},shared_encoder_n_layers = 1,trainer=True,
    early_stopping_patience=2,early_stopping_metric=None,pretrained_model='distilroberta-base',frozen_layers='all',scoring='script',
    normalised_values=None,save_results=True
    ):
    if type(init_task_weightings)==int or type(init_task_weightings)==float:
        init_task_weightings = calculate_init_loss_hyper_params(init_task_weightings,tasks,task_priorities)

    map_task_index = {task:i for i,task in enumerate(tasks)}
    map_out_to_task={f'{task}_{output}':task  for task in tasks for output in outputs}
    map_out_to_out={f'{task}_{output}':output  for task in tasks for output in outputs}
    task_priorities_priority_as_key = {v:k for k,v in task_priorities.items()}
    kwargs = {
        'tasks':tasks,
        'outputs':outputs,
        'map_out_to_task':map_out_to_task,
        'map_out_to_out':map_out_to_out,
        'map_task_index':map_task_index,
        'map_index_task':{v:k for k,v in map_task_index.items()},
        'optimizer_weighting':optimizer_weighting,
        'epochs_to_avg_over':2,
        'temp':temp,
        'init_task_weightings':init_task_weightings,
        'metrics_to_track_by_task':metrics_to_track_by_task,
        'regression_tasks':regression_tasks,
        'classification_tasks':classification_tasks,
        'fce_tasks':fce_tasks,
        'task_priorities_priority_as_key':task_priorities_priority_as_key,
        'early_stopping_patience':early_stopping_patience,
        'early_stopping_metric':early_stopping_metric,
        'pretrained_model':pretrained_model,
        'frozen_layers':frozen_layers,
        'scoring':scoring,
        'shared_encoder_n_layers':shared_encoder_n_layers,
        # normalising score by multiplying by the first normalisation value and adding the second
        'normalised_values':normalised_values,
        'output_layer_by_task':output_layer,
        'save_results':save_results
    }
    task_dict = {
        task:{
          'class_weights':class_weights[task],
          'n_labels':n_labels[task],
          'labels':labels[task],
          'model':model[task],
          'shares_encoder':shares_encoder[task],
          'output_layer':output_layer[task]
          } 
        for task in tasks
        }
    task_dict['shared_encoder']=True
    task_dict['fce_tasks'] = fce_tasks
    return kwargs,{**task_dict,**kwargs}

#caclulating fixed weights for the taskbased on the priority of the task 
def calculate_init_loss_hyper_params(primary_task_weight,tasks,task_priorities):
    updated_init_task_weightings = {}
    for task in tasks:
        if task_priorities[task]=='primary_task':
            updated_init_task_weightings[task] = primary_task_weight
        elif task_priorities[task]=='secondary_task':
            updated_init_task_weightings[task] = 1-primary_task_weight
        elif task_priorities[task]=='aux_task':
            updated_init_task_weightings[task] = (1-primary_task_weight)*0.1
    return updated_init_task_weightings

# determine the dataset needed by task and model type
def get_dataset(tasks,pretrained_model='distilroberta-base',max_length=512):
    dataset_obj = CreateHuggingFaceMultiTask(pretrained_model=pretrained_model,max_length=max_length)
    if ('aes' in tasks and 'ner' in tasks) or ('ged' in tasks and 'ner' in tasks):
        dataset_dict = dataset_obj.get_combined_dataset_dict()
    elif ('ner' in tasks) and ('aes' not in tasks and 'ged' not in tasks):
        if 'bert-' in pretrained_model:
            dataset_dict = dataset_obj.get_ner_dataset_dict_primary()
        else:
            dataset_dict = dataset_obj.get_ner_dataset_dict()
    else:
        dataset_dict = dataset_obj.get_fce_dataset_dict()
    return dataset_obj,dataset_dict

# freeze layers of a model
def freeze_layers(model,frozen_layers='all',model_type='roberta'):
    try:
        if frozen_layers=='all':
            for name,params in model.model.named_parameters():
                if 'classifier' not in name:
                    params.requires_grad = False
            return model
    
        elif type(frozen_layers)==int:
            if model_type == 'roberta':
                modules = [model.model.roberta.embeddings, *model.model.roberta.encoder.layer[:frozen_layers]] 
                for module in modules:
                    for param in module.parameters():
                        param.requires_grad = False
            elif model_type == 'bert':
                modules = [model.model.bert.embeddings, *model.model.bert.encoder.layer[:frozen_layers]] 
                for module in modules:
                    for param in module.parameters():
                        param.requires_grad = False
            return model
        else:
            return model
    except:
        modules = [model.model.embeddings, *model.model.encoder.layer[:frozen_layers]] 
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        return model

def normalise_scores(example):
    score = example['scores']/40
    return {'scores':score}

def create_params_grid(params):
    return list(ParameterGrid(params))
