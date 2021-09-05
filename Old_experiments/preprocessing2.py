
import pandas as pd
import numpy as np
import pdb
import os
os.chdir('/content/drive/MyDrive/CAMemBERT2')
import re
from datasets import Dataset,DatasetDict,load_dataset,concatenate_datasets
from transformers import AutoTokenizer
from torch import FloatTensor
from torch.cuda import is_available
from math import ceil,floor
import random

class LinkGedDatasetToEssayDataset:

    _map_labels_2_ids = {'c':0,'i':1}
    _map_ids_2_labels = {0:'c',1:'i'}

    def __init__(self,set_type='train'):
        self.errors_count=0
        self.set_type = 'dev'
        essays = pd.read_json(f'data/fce.{set_type}.json',lines=True)
        essays['text'] = essays.text.str.replace('\n',' ')
        essays['text_no_ws'] = essays.text.str.split().str.join('')
        essays['essay_char_len'] = essays['text_no_ws'].apply(len)
        essays['end_word_ind'] = essays['essay_char_len'].cumsum()
        essays['start_ind'] = essays['end_word_ind'] - essays['essay_char_len']
        essays['ind_combined'] = essays.apply(lambda x: list([x['start_ind'],x['end_word_ind']]),axis=1)
        # essays['text_no_ws_from'] = essays['ind_combined'].apply(lambda x:''.join(df.loc[(df['end_word_ind']>x[0])&(df['end_word_ind']<=x[1])].word.tolist()))
        self.essays = essays
        self.essay_col_index = {col:i+1 for i,col in enumerate(essays)}
        ged = pd.DataFrame(pd.read_csv(f'data/fce-public.{set_type}.original.tsv',sep='  ',names=['word']).word.str.split('\t',1).tolist(),columns = ['word','correct'])
        ged['correct'] = ged['correct'].map(self._map_labels_2_ids)
        ged['end_word_ind'] = ged.word.apply(len).cumsum()
        self.ged = ged
        self.all_words_no_ws = ''.join(ged.word.tolist())
        if set_type=='test':
            essay_indexes_to_keep,matched_essays = zip(*[(i,re.search(re.escape(ess_no_ws),self.all_words_no_ws)) for i,ess_no_ws in enumerate(essays.text_no_ws.tolist()) if ess_no_ws in self.all_words_no_ws ])
            essays_from_ged = [' '.join(self.ged.loc[(self.ged['end_word_ind']>m.start())& (self.ged['end_word_ind']<=m.end())]['word'].tolist()) for m in matched_essays]
            tags_from_ged = [self.ged.loc[(self.ged['end_word_ind']>m.start())& (self.ged['end_word_ind']<=m.end())]['correct'].tolist() for m in matched_essays]
            essays = essays.iloc[list(essay_indexes_to_keep)].reset_index()
        else:
            self.find_differences()
            essays_from_ged = [' '.join(self.ged.loc[(self.ged['end_word_ind']>ind[0])& (self.ged['end_word_ind']<=ind[1])]['word'].tolist()) for ind in self.new_indexes]
            tags_from_ged = [self.ged.loc[(self.ged['end_word_ind']>ind[0])& (self.ged['end_word_ind']<=ind[1])]['correct'].tolist() for ind in self.new_indexes]
        self.updated_df = pd.concat([pd.DataFrame({'essays':essays_from_ged,'tags':tags_from_ged}),essays[['answer-s','script-s','id']]],axis=1)

    def get_updated_df(self):
        return self.updated_df

    def find_differences(self):
        self.new_text = []
        self.new_indexes = []
        self.new_indexes_errors = []
        for i,row in enumerate(self.essays.itertuples()):
            essay_errors_count = 0
            essay_no_ws = row[self.essay_col_index['text_no_ws']]
            char_len = row[self.essay_col_index['essay_char_len']]
            start_ind,end_ind = row[self.essay_col_index['start_ind']]-self.errors_count,row[self.essay_col_index['end_word_ind']]-self.errors_count
            ged_essay = self.all_words_no_ws[start_ind:end_ind]
            self.new_indexes.append([start_ind,end_ind])
            try:
                np.all(np.array(list(essay_no_ws))==np.array(list(ged_essay)))
            except:
                if self.set_type=='dev':
                    essay_no_ws = essay_no_ws.replace("''","")
            if np.all(np.array(list(essay_no_ws))==np.array(list(ged_essay))):
                self.new_text.append(essay_no_ws)
            else:
                
                current_error = min(np.nonzero(np.invert(np.array(list(essay_no_ws))==np.array(list(ged_essay))))[0])
                if np.all(essay_no_ws[current_error+1:char_len] == ged_essay[current_error:char_len-1]):
                    tmp_text = essay_no_ws[current_error+1:char_len]
                    combined_text = essay_no_ws[:current_error]+tmp_text
                    self.new_text.append(combined_text)
                    self.errors_count+=1
                else:
                    correct_text = ged_essay[:current_error]
                    a = True
                    while a == True:
                        tmp_ess = essay_no_ws[current_error+essay_errors_count+1:char_len]
                        tmp_ged = ged_essay[current_error:char_len-essay_errors_count-1]
                        if tmp_ess == tmp_ged:
                            tmp_text = essay_no_ws[current_error+1:char_len]
                            correct_text = correct_text + tmp_ged
                            self.new_text.append(correct_text)
                            self.errors_count+=essay_errors_count
                            a = False
                            break

                        else:
                            char_to_next_error = min(np.nonzero(np.invert(np.array(list(tmp_ess))==np.array(list(tmp_ged))))[0])
                            current_error += char_to_next_error
                            correct_text = correct_text + tmp_ged[:char_to_next_error]
                            essay_errors_count+=1

class CreateHuggingFaceDictGed:

    _set_types = ['train','test','dev']
    _cols_to_keep = ['attention_mask','labels','input_ids','script_scores']

    def __init__(self,pretrained_model= 'distilroberta-base',max_length=512):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        combined_df = pd.concat([self.add_col(LinkGedDatasetToEssayDataset(set_type).get_updated_df(),'set_type',set_type) for set_type in self._set_types],axis=0)
        df = combined_df.rename(columns={'essays':'text','script-s':'script_scores','tags':'labels'})
        df['script_scores'] = df['script_scores'].astype(int)        
        df = df.groupby('id').agg({'text':list,'labels':list,'script_scores':list,'set_type':list})
        df[ 'text' ] = df[ 'text' ].str.join(' ')
        df['labels'] = df['labels'].apply(lambda x : x[0] + x[1] if len(x)>1 else x[0])
        df[ 'script_scores' ] = df[ 'script_scores' ].apply(lambda x : x[0])
        df[ 'set_type' ] = df[ 'set_type' ].apply(lambda x : x[0])
        df = df.reset_index()[['text','labels','script_scores','set_type']]
        self.df_for_sent = df
        dataset_dict = DatasetDict({set_type:Dataset.from_pandas(df.groupby('set_type').get_group(set_type)) for set_type in self._set_types})
        dataset_dict = dataset_dict.map(self.extend_labels_for_tokenizer).map(self.preprocessing_func)
        cols_to_drop = set(dataset_dict.column_names['train']) - set(self._cols_to_keep)
        self.dataset_dict = dataset_dict.remove_columns(list(cols_to_drop))
        self.set_weights()

    def add_col(self,df,col,val):
        df[col] = val
        return df

    def get_df(self):
        return self.df_for_sent

    def get_dataset_dict(self):
        return self.dataset_dict

    def get_weights(self):
        return self.class_weights

    def extend_labels_for_tokenizer(self,example):
        tokens,labels = example['text'].split(),example['labels']
        r_tags , token2word = [] , []
        count = 0
        for index, token in enumerate( self.tokenizer.tokenize( ' '.join( tokens ) , truncation = True , padding = False , add_special_tokens = False , max_length = self.max_length ) ):

            if ( ( ( ( token.startswith( "Ġ" ) == False and index != 0 ) or ( token in tokens[ index - count - 1 ].lower() and index - count - 1 >= 0 ) ) and self.tokenizer.sep_token == '</s>' ) 
                or ( ( token.startswith( "##" ) or ( token in tokens[index - count - 1].lower() and index - count - 1 >= 0 ) ) and self.tokenizer.sep_token == '[SEP]' ) ):

                r_tags.append( -100 )
                
                count += 1

            else:

                try:
                    r_tags.append(labels[index - count])
                except:
                    pdb.set_trace()

            token2word.append( index - count )
        return {'labels':np.pad( r_tags , ( 0 , 512 - len( r_tags ) ) , 'constant' , constant_values = ( 0 , -100 ) )[:self.max_length]}

    def preprocessing_func(self,example):
        return self.tokenizer( example['text'] , truncation=True , padding = 'max_length' , max_length = self.max_length )

    def set_weights(self):
        dataset = self.get_dataset_dict()
        padding,n_c,n_i = np.unique(np.concatenate(dataset['train']['labels']),return_counts=True)[1]
        class_weights = FloatTensor([(n_c + n_i)/(2.0 * n_c),(n_c + n_i)/(2.0 * n_i)]).to('cuda' if is_available() else 'cpu')
        self.class_weights = class_weights


class CreateHuggingFaceDictNer:

    _set_types = ['train','test','validation']
    _cols_to_keep_before_dataset_conversion = ['tokens','ner_tags']

    def __init__(self,pretrained_model= 'distilroberta-base',max_length=512, ged_obj=None):
        if ged_obj==None:
            self.ged_obj=CreateHuggingFaceDictGed(pretrained_model,max_length)
        else:
            self.ged_obj=ged_obj
        self.ged_dataset = ged_obj.get_dataset_dict()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        dataset = load_dataset("conll2003")
        self.dataset = dataset
        dataset_dict = self.reshape_to_match_to_essays_dataset(dataset)
        dataset_dict = dataset_dict.map(self.ged_obj.extend_labels_for_tokenizer).map(self.ged_obj.preprocessing_func).remove_columns(['text'])
        self.dataset_dict = dataset_dict

    def get_dataset_dict(self):
        return self.dataset_dict

    def get_dataset(self):
        return self.dataset

    def append_sep_and_pad(self,example):
        tokens,tags = example['tokens'],example['ner_tags']
        tokens.append(self.tokenizer.sep_token)
        tags.append(-100)
        return {'tokens':tokens,'ner_tags':tags}

    def reshape_to_match_to_essays_dataset(self,dataset):
        dataset_dict = {}
        self.max_length_for_training = None
        for set_type,set_type_ged in zip(self._set_types,self.ged_obj._set_types):
            type_data = []
            for col in self._cols_to_keep_before_dataset_conversion:
                flattened_values = np.concatenate(dataset[set_type][col])
                if set_type=='train':
                    split_arr = np.array_split(flattened_values,self.ged_dataset[set_type_ged].num_rows)
                    if self.max_length_for_training==None:
                        self.max_length_for_training = max([len(row) for row in split_arr])
                else:
                    length_of_new_arr = ceil(len(flattened_values)/self.max_length_for_training)
                    split_arr = np.array_split(flattened_values,length_of_new_arr)

                

                # split_arr = np.array_split(flattened_values,self.ged_dataset[set_type_ged].num_rows)
                # max_len_row = max([len(row) for row in split_arr])
                # self.max_length_by_set_type[set_type] = max_len_row
                padded_array = []
                for row in split_arr:
                    if col == 'tokens':
                        padded_array.append(' '.join(list(np.pad(row,(0,self.max_length_for_training-len(row)),constant_values = (self.ged_obj.tokenizer.pad_token,self.ged_obj.tokenizer.pad_token)))))
                    else:
                        padded_array.append(list(np.pad(row,(0,self.max_length_for_training-len(row)),constant_values = (-100,-100))))


                # padded_array = []
                # for row in split_arr:
                #     if col == 'tokens':
                #         padded_array.append(' '.join(list(np.pad(row,(0,max_len_row-len(row)),constant_values = (self.ged_obj.tokenizer.pad_token,self.ged_obj.tokenizer.pad_token)))))
                #     else:
                #         padded_array.append(list(np.pad(row,(0,max_len_row-len(row)),constant_values = (-100,-100))))
                type_data.append(pd.Series(padded_array))
            tmp_df =pd.DataFrame({'text':type_data[0],'labels':type_data[1]})
            dataset_dict[set_type] = Dataset.from_pandas(tmp_df)
        return DatasetDict(dataset_dict)


class CreateHuggingFaceMultiTask:

    _col_to_add_to_ner = ['script_score']
    _set_types = ['train','test','dev']
    _map_ged_set_to_ner = {'train':'train','test':'test','dev':'validation'}
    _map_ner_set_to_ged = {'train':'train','test':'test','validation':'dev'}
    _tasks = ['ged','ner']

    def __init__(self,pretrained_model='distilroberta-base',max_length=512,ged_obj=None,ner_obj=None, batch_size=8):
        if ged_obj==None:
            self.ged_obj=CreateHuggingFaceDictGed(pretrained_model,max_length)
        else:
            self.ged_obj=ged_obj
        if ner_obj==None:
            self.ner_obj=CreateHuggingFaceDictNer(pretrained_model,max_length,self.ged_obj)
        else:
            self.ner_obj=ner_obj
        self.ged_dataset_dict = self.ged_obj.get_dataset_dict()
        self.ner_dataset_dict = self.ner_obj.get_dataset_dict()
        self.batch_size = batch_size
        for set_type in self.ner_obj._set_types:
            self.ner_dataset_dict[set_type] =  self.ner_dataset_dict[set_type].add_column('script_scores',[-100]*self.ner_dataset_dict[set_type].num_rows)
            self.ner_dataset_dict[set_type] = self.ner_dataset_dict[set_type].cast(self.ged_dataset_dict[self._map_ner_set_to_ged[set_type]].features)
        self.dataset_dict = self.combine_datasets()

    def get_ged_datset(self):
        return self.ged_dataset_dict

    def get_ner_datset(self):
        return self.ner_dataset_dict

    def get_combined_dataset(self):
        return self.dataset_dict

    def generate_concatenated_loader(self):
        dataset_lst = [self.get_ged_datset(),self.get_ner_datset()]
        concatenated_datasets = concatenate_datasets(dataset_lst)
        lengths = [dset.num_rows for dset in dataset_lst]
        offsets = np.cumsum([0] + lengths[:-1])
        indices = (offsets.reshape(1, -1) + np.arange(max(lengths)).reshape(-1, 1)).flatten().tolist()
        return concatenated_datasets.select(indices)

    def combine_datasets(self):
        dataset_dict = {}
        for set_type in self._set_types:
            dataset_lst = [self.get_ged_datset()[set_type],self.get_ner_datset()[self._map_ged_set_to_ner[set_type]]]
            concatenated_datasets = concatenate_datasets(dataset_lst)
            if set_type!='train':
                dataset_dict[set_type] = concatenated_datasets
            else:
                lengths = [dset.num_rows for dset in dataset_lst]
                offsets = np.cumsum([0] + lengths[:-1])
                indexes = list(np.arange(min(lengths)))
                indicies = [offset + indexes for offset in offsets]
                batch_order=[]
                for _ in range(ceil(min(lengths)/self.batch_size)):
                    try:
                        samples = random.sample(indexes , self.batch_size)
                        batch_order.append(samples)
                        indexes = [ind for ind in indexes if ind not in samples]
                    except:
                        batch_order.append(indexes)
                bath_indexes = [[ind[mini_batch_inds] for ind in indicies] for mini_batch_inds in batch_order]
                batches_flattened = [list(mini_batch) for mini_batches in bath_indexes for mini_batch in mini_batches]
                dataset_dict[set_type] = concatenated_datasets.select( np.concatenate(batches_flattened) )

        return DatasetDict(dataset_dict)








