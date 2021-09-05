# imports
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
import json
import time

# A class that merges the fce grammatical error detection (ged) dataset to fce automated essay scoring dataset (aes) 
# so that grammar tags and essay scores can be used.
# for further notes of how this class works and additioanl essay matching methods see bottom
class LinkGedDatasetToEssayDataset:

    # mapping for string labels to numerical ones in ged dataset
    _map_labels_2_ids = {'c':0,'i':1}

    def __init__(self,set_type='train'):
        # counter for the number of differences where characters appear in the aes dataset but not ged dataset
        self.errors_count=0
        # set type can be train, test, dev
        self.set_type = set_type
        self.essays = self.read_and_parse_essay_data()
        # create mapping of column. names to indexes for itertuples iteration in self.get_differences() method
        self.essay_col_index = {col:i+1 for i,col in enumerate(self.essays)}
        self.ged = self.read_and_parse_ged_data()
        # string containing all words in ged dataset with no whitespace
        self.all_words_no_ws = ''.join(self.ged.word.tolist())
        # final dataframe with essay text, essay scores, essay ids and essay grammar labels 
        self.updated_df = self.match_essays_and_grammar_labels()

    def get_updated_df(self):
        return self.updated_df

    def read_and_parse_essay_data(self):
        essays = pd.read_json(f'data/fce.{self.set_type}.json',lines=True)
        essays['text'] = essays.text.str.replace('\n',' ')
        essays['text_no_ws'] = essays.text.str.split().str.join('')
        essays['essay_char_len'] = essays['text_no_ws'].apply(len)
        essays['end_word_ind'] = essays['essay_char_len'].cumsum()
        essays['start_ind'] = essays['end_word_ind'] - essays['essay_char_len']
        essays['ind_combined'] = essays.apply(lambda x: list([x['start_ind'],x['end_word_ind']]),axis=1)
        return essays

    def read_and_parse_ged_data(self):
        ged = pd.DataFrame(pd.read_csv(f'data/fce-public.{self.set_type}.original.tsv',sep='  ',names=['word']).word.str.split('\t',1).tolist(),columns = ['word','labels'])
        ged['labels'] = ged['labels'].map(self._map_labels_2_ids)
        ged['end_word_ind'] = ged.word.apply(len).cumsum()
        return ged

    def match_essays_and_grammar_labels(self):
        # essays_to_keep = index of all essays that form an exact match with all words in the ged dataset string
        essays_to_keep,matched_essays = zip(*[(i,re.search(re.escape(essay_no_ws),self.all_words_no_ws)) for i,essay_no_ws 
                                                      in enumerate(self.essays['text_no_ws'].tolist()) 
                                                      if essay_no_ws in self.all_words_no_ws ])
        # get the words from the ged dataset (with whitespace) to form each essay
        essays_from_ged = [' '.join(self.ged.loc[(self.ged['end_word_ind']>m.start())& (self.ged['end_word_ind']<=m.end())]['word'].tolist()) for m in matched_essays]
        # get the labels from the ged dataset corresponding to each essay
        labels_from_ged = [self.ged.loc[(self.ged['end_word_ind']>m.start())& (self.ged['end_word_ind']<=m.end())]['labels'].tolist() for m in matched_essays]

        # make sure essays dataframe only contains essays that have matched
        essays = self.essays.iloc[list(essays_to_keep)].reset_index()
        return pd.concat([pd.DataFrame({'text':essays_from_ged,'labels':labels_from_ged}),essays[['answer-s','script-s','id']]],axis=1)


# Class to create a huggingface FatasetDict for the fce dataset for tasks of aes and ged
class CreateHuggingFaceDictFce:

    # class varaibles 
    # possible set types
    _set_types = ['train','dev','test']
    _cols_to_keep = ['attention_mask','labels','input_ids','scores']
    _answer_score_mapping = {
                      0.0:0,
                      1.1:1,1.2:4,1.3:8,
                      2.1:9,2.2:10,2.3:11,
                      3.1:12,3.2:13,3.3:14,
                      4.1:15,4.2:16,4.3:17,
                      5.1:18,5.2:19,5.3:20,
                  }

    def __init__(self,pretrained_model= 'distilroberta-base',max_length=512,scoring='script'):
        # max length for tokenization
        self.max_length = max_length
        # huggingface tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.scoring = scoring
        self.fce_df = self.parse_data()
        self.dataset_dict = self.create_hf_dataset_dict()
        self.set_weights()

    def get_df(self):
        return self.df_for_sent

    def get_dataset_dict(self):
        return self.dataset_dict

    def get_weights(self):
        return self.class_weights

    def parse_data(self):
        # create one dataframe containing all samples from all train, test and dev set 
        combined_df = pd.concat([self.add_col(LinkGedDatasetToEssayDataset(set_type).get_updated_df(),'set_type',set_type) for set_type in self._set_types],axis=0)
        if self.scoring == 'script':
            df = combined_df.rename(columns={'script-s':'scores'})
            df['scores'] = df['scores'].astype(float)
            # group all essays by id and combine the text and labels as well as merge the scores and set types        
            df = df.groupby('id').agg({'text':list,'labels':list,'scores':list,'set_type':list})
            df[ 'text' ] = df[ 'text' ].str.join(' ')
            df['labels'] = df['labels'].apply(lambda x : x[0] + x[1] if len(x)>1 else x[0])
            df[ 'scores' ] = df[ 'scores' ].apply(lambda x : x[0])
            df[ 'set_type' ] = df[ 'set_type' ].apply(lambda x : x[0])
        elif self.scoring == 'answer':
            df = combined_df.rename(columns={'answer-s':'scores'})
            # correct for errors in scoring
            df[ 'scores' ] = df[ 'scores' ].str.replace( '/','.' ).str.replace('T','')
            # remove values containing non-numeric data and are in score mappings
            df = df[ ~pd.to_numeric( df[ 'scores' ] , errors='coerce' ).isna() ]
            df = df[ df['scores'].astype(float).isin(self._answer_score_mapping.keys())]
            # map scores to new values
            df[ 'scores' ] = df[ 'scores' ].astype(float).map(self._answer_score_mapping)
            # return only certain columns from the original dataset
        return df.reset_index()[['text','labels','scores','set_type']]

    # used to add set type column to each of the train test and dev dataframes
    def add_col(self,df,col,val):
        df[col] = val
        return df

    def create_hf_dataset_dict(self):
        # create a hugging face dataset for each of the train test and dev samples and combine them to create a huggingface dataset dictionary
        dataset_dict = DatasetDict({set_type:Dataset.from_pandas(self.fce_df.groupby('set_type').get_group(set_type)) for set_type in self._set_types})
        # apply the method to extend the labels for grammatical error detection and tokenize each essay
        dataset_dict = dataset_dict.map(self.extend_labels_for_tokenizer).map(self.tokenize_text)
        # find the columns to drop from the new dataset dict
        cols_to_drop = set(dataset_dict.column_names['train']) - set(self._cols_to_keep)
        return dataset_dict.remove_columns(list(cols_to_drop))

    def extend_labels_for_tokenizer(self,example):
        # split text by white space to create individual tokens that correspond to labels
        tokens,labels = example['text'].split(),example['labels']
        labels_for_tokens = [] 
        # split word counts (words that are plit up by tokenizer)
        split_count = 0
        # iterate through each token generated when the tokenizer is applied to the full length text
        for index, token in enumerate( self.tokenizer.tokenize( ' '.join( tokens ) , truncation = True , padding = False , add_special_tokens = False , max_length = self.max_length ) ):
            # if conditions to determine if the tokenizer has split a word based on tokenizer used
            if ( ( ( ( token.startswith( "Ġ" ) == False and index != 0 ) or ( token in tokens[ index - split_count - 1 ].lower() and index - split_count - 1 >= 0 ) ) and self.tokenizer.sep_token == '</s>' ) 
                or ( ( token.startswith( "##" ) or ( token in tokens[index - split_count - 1].lower() and index - split_count - 1 >= 0 ) ) and self.tokenizer.sep_token == '[SEP]' ) ):
                # add a padding token for words that are split by the tokenizer
                labels_for_tokens.append( -100 )
                # add a count 
                split_count += 1
            else:
                # add the label to all tokens that either haven't been split by the tokenizer or are the first word of a split
                labels_for_tokens.append(labels[index - split_count])
        # pad and truncate the labels to be the max length of the tokenizer by padding -100 to the token length where necessary
        return {'labels':np.pad( labels_for_tokens , ( 0 , self.max_length - len( labels_for_tokens ) ) , 'constant' , constant_values = ( 0 , -100 ) )[:self.max_length]}

    # get the padded and truncated input ids and attention masks for each text (essay)
    def tokenize_text(self,example):
        return self.tokenizer( example['text'] , truncation=True , padding = 'max_length' , max_length = self.max_length )

    # set weights to apply to the cross entropy loss function to penalise for under represented classes
    def set_weights(self):
        dataset = self.get_dataset_dict()
        padding,n_c,n_i = np.unique(np.concatenate(dataset['train']['labels']),return_counts=True)[1]
        class_weights = FloatTensor([(n_c + n_i)/(2.0 * n_c),(n_c + n_i)/(2.0 * n_i)]).to('cuda' if is_available() else 'cpu')
        self.class_weights = class_weights

# Class to create a huggingface DatasetDict for the conll2003 dataset for task of Named Entity Recognition NER
class CreateHuggingFaceDictNerandAesDataset(CreateHuggingFaceDictFce):

    # class variables
    # possible set types
    _cols_to_keep_before_dataset_conversion = ['tokens','ner_tags']
    ner_dataset = "conll2003"

    def __init__(self,pretrained_model= 'distilroberta-base',max_length=512,scoring='script'):
        super().__init__(pretrained_model,max_length,scoring)
        # load in ner dataset from huffingface
        dataset = load_dataset(self.ner_dataset).map(self.append_sep_and_pad)
        self._set_types_ner = list(dataset.keys())
        dataset_dict = self.reshape_to_match_to_fce_dataset(dataset)
        # apply tokenization
        dataset_dict = dataset_dict.map(self.extend_labels_for_tokenizer).map(self.tokenize_text).remove_columns(['text'])
        self.dataset_dict_ner = dataset_dict

    def get_ner_dataset_dict(self):
        return self.dataset_dict_ner

    def get_fce_dataset_dict(self):
        return self.dataset_dict

    # append a sep token to the end of each sample from the ner dataset and a padtoken to the end of each label
    def append_sep_and_pad(self,example):
        tokens,tags = example['tokens'],example['ner_tags']
        tokens.append(self.tokenizer.sep_token)
        tags.append(-100)
        return {'tokens':tokens,'ner_tags':tags}

    def reshape_to_match_to_fce_dataset(self,dataset):
        dataset_dict = {}
        # variable that dictates the length of tokenization in the ner dataset so that the number of samples/rows match that of the fce dataset
        self.max_length_for_training = None
        # iterate through the set types of the ner and fce datasets
        for set_type,set_type_fce in zip(self._set_types_ner,self._set_types):
            # list_for_each_set_types_data
            type_data = []
            # iterate through each column that is needed from the ner dataset
            for col in self._cols_to_keep_before_dataset_conversion:
                # flatten all the values in the curent column to a 1d array
                flattened_values = np.concatenate(dataset[set_type][col])
                # if the set type is train clalculate the length of tokenization required to ensure the number of rows in the ner dataset 
                # is equal to the number of rows in the fce
                if set_type=='train':
                    split_arr = np.array_split(flattened_values,self.get_fce_dataset_dict()[set_type_fce].num_rows)
                    if self.max_length_for_training==None:
                        max_row_length_of_split_array =  max([len(row) for row in split_arr])
                        self.max_length_for_training = max_row_length_of_split_array if max_row_length_of_split_array <= self.max_length else self.max_length
                # if the dataset is not train then set the dataset to have the same max length tokenization as max_length_for_training
                # unless max length for training is greater than max length of tokenizer in which case the max_length of tokenizer should be used
                else:
                    # reshape array to have the correct max length of tokenization but an unlimited number of rows
                    length_of_new_arr = ceil(len(flattened_values)/self.max_length_for_training )
                    split_arr = np.array_split(flattened_values,length_of_new_arr)

                # pad each row in the split array to have the same length
                padded_array = []
                for row in split_arr:
                    if col == 'tokens':
                        padded_array.append(' '.join(list(np.pad(row,(0,self.max_length_for_training -len(row)),constant_values = (self.tokenizer.pad_token,self.tokenizer.pad_token))[:self.max_length_for_training])))
                    else:
                        padded_array.append(list(np.pad(row,(0,self.max_length_for_training -len(row)),constant_values = (-100,-100))[:self.max_length_for_training ]))
                type_data.append(pd.Series(padded_array))
            # create a hugging face dataset from the set type
            tmp_df = pd.DataFrame({'text':type_data[0],'labels':type_data[1]})
            dataset_dict[set_type] = Dataset.from_pandas(tmp_df)
        # combine all data into datasetdict
        return DatasetDict(dataset_dict)

# to note:
#           - ner training dataset is padded to be the same width and length as the fce dataset so that it can be loaded into a model without the need for over
#             under sampling. 
class CreateHuggingFaceMultiTask(CreateHuggingFaceDictNerandAesDataset):

    # class variables
    _col_to_add_to_ner = ['score']
    _set_types = ['train','test','dev']
    _map_fce_set_to_ner = {'train':'train','test':'test','dev':'validation'}
    _map_ner_set_to_fce = {'train':'train','test':'test','validation':'dev'}
    _tasks = ['aes','ged','ner']

    def __init__(self,pretrained_model='distilroberta-base',max_length=512, scoring='script',batch_size=8):
        super().__init__(pretrained_model,max_length, scoring)
        self.fce_dataset_dict = self.get_fce_dataset_dict()
        self.ner_dataset_dict = self.get_ner_dataset_dict()
        self.batch_size = batch_size
        if 'token_type_ids' in self.ner_dataset_dict.column_names['train']:
            self.ner_dataset_dict = self.ner_dataset_dict.remove_columns('token_type_ids')

        for set_type in self._set_types:
            # add a dataset column to fce dataset
            self.fce_dataset_dict[set_type] = self.fce_dataset_dict[set_type].add_column('dataset',[0]*self.fce_dataset_dict[set_type].num_rows)
        for set_type in self._set_types_ner:
            # add a score column to ner dataset and pad values 
            self.ner_dataset_dict[set_type] = self.ner_dataset_dict[set_type].add_column('scores',[-100]*self.ner_dataset_dict[set_type].num_rows)
            # add a dataset column to ner dataset
            self.ner_dataset_dict[set_type] = self.ner_dataset_dict[set_type].add_column('dataset',[1]*self.ner_dataset_dict[set_type].num_rows)
            # make sure all columns are of the same dataset type
            self.ner_dataset_dict[set_type] = self.ner_dataset_dict[set_type].cast(self.fce_dataset_dict[self._map_ner_set_to_fce[set_type]].features)
        # combine the two datasets through concatanation
        self.combined_dataset_dict = self.combine_datasets()

    def get_combined_dataset_dict(self):
        return self.combined_dataset_dict

    def get_ner_dataset_dict_primary(self):
        ner = self.get_ner_dataset_dict()
        for set_type in self._set_types_ner:
            if 'dataset' not in ner.column_names[set_type]:
                ner[set_type] = ner[set_type].add_column('dataset',[1]*ner[set_type].num_rows)
        return ner

    # generate a dataloader for training and testing models so that data is loaded in alternating tasks for training 
    # and all samples from one task then another for testing and dev.
    def combine_datasets(self):
        dataset_dict = {}
        for set_type in self._set_types:
            # concatanate datasets so one follows another from a list of all the datasets
            dataset_lst = [self.fce_dataset_dict[set_type],self.ner_dataset_dict[self._map_fce_set_to_ner[set_type]]]
            concatenated_datasets = concatenate_datasets(dataset_lst)
            if set_type=='train':
                # get the length of each dataset
                lengths = [dset.num_rows for dset in dataset_lst]
                # get the offset for each dataset (number of samples between the begining of the concatanated dataset and the start of a new dataset)
                offsets = np.cumsum([0] + lengths[:-1])
                # get a list of indexes for the minimum length dataset (although both the same length)
                indexes = list(np.arange(min(lengths)))
                # get a list of all the possible indexes in the smallest / first dataset
                indicies = [offset + indexes for offset in offsets]
                # list for storing the order which batches should appear in
                batch_order=[]
                for _ in range(ceil(min(lengths)/self.batch_size)):
                    # create a list of mini batch indexes by appending randomly sampled indexes of length batch size 
                    # until they run out / can no longer fill a batch and then append the remaining to the last batch
                    # (this is samples only for the smallest / first dataset)
                    try:
                        samples = random.sample(indexes , self.batch_size)
                        batch_order.append(samples)
                        indexes = [ind for ind in indexes if ind not in samples]
                    except:
                        batch_order.append(indexes)
                # extend samples to both datasets
                bath_indexes = [[ind[mini_batch_inds] for ind in indicies] for mini_batch_inds in batch_order]
                # flatten out the list of lists (potential for one mixed batch that will be handled by model)
                batches_flattened = [list(mini_batch) for mini_batches in bath_indexes for mini_batch in mini_batches]
                # select samples in order defined above
                dataset_dict[set_type] = concatenated_datasets.select( np.concatenate(batches_flattened) )
            else:
                dataset_dict[set_type] = concatenated_datasets
        return DatasetDict(dataset_dict)



 


# further notes:
#           - each row in the ged dataset represents one token and its corresponding error label, 
#             where as one row in the aes dataset represents one essay and its score.
#           - each token has been formed using rasp tokenization.
#           - words in ged dataset appear in the same order as they do in essay dataset.
#           - some essays in the ged dataset contain extra words or miss words when compared to the original dataset.
#             these are omitted from the dataset. 
# Works by: 
#           - indexing the end of every word in the ged dataset
#           - joining all the words in the ged dataset together with no whitespace 
#             (creating one string of all the words in the ged dataset (referred to as ged text))
#           - joining all the words in each individual essay together with no whitespace
#           - locating where the essay with no whitespace matches the sequence of words in the joined ged text 
#           - use the index of the start and end word of the appearance of the essay in the ged text as a way to locate the rows in the ged dataset 
#             corresponding to the words in an essay, these rows to get the error labels and words for each essay and merged with essay score and grammar labels.
# Further notes:
#           - however, this does mean the tokenization used to split up words in the essay by the ged dataset does affect the final appearance of the essay.
#             (as more whitespace appears due to splitting of individual words by the tokenizer in the ged dataset), which was seen to negatively impact essay predictions.
#           - but, using the original essay leads to incorrect tagging of the words in each essay; as a result of labels needing to be extended for transformer tokenization.
#             (which was seen to negatively impact grammar predictions).
#           - the impact of incorrect tagging was considdered to be of great impact to the overall validity of the model, hence the joined words from the ged
#             were used as opposed to using the original words dataset.
#           - attempts at making the essays and tags match exactly by locating differences in between the two datasets were somewhat but were at risk of error
#             so there were 70 scripts omitted from the original training set, 7 from the original developement set and 9 from the original test set.
#             Additionally, attempts to perform exact matches were futile for the test dataset as essays in the aes dataset appeared in a different order to the ged dataset. 

# Following code block omitted from experimentation due to possible errors
    ###################################################################################################################################
# class LinkGedDatasetToEssayDatasetWithAdditionalMatching(LinkGedDatasetToEssayDataset):

#     def __init__(self,set_type='train'):
#       super.__init__(set_type)

#     def additional_matching(self):
#         # create a list of essays that match between the essay and grammar data by using the start and end index of the essay 
#         # as they appear in the aes dataset.
#         # done by : using the cumulative sum of the length of each essay to get indexes in the aes dataset.
#         # then using the indexes to find the differences where characters appear in the aes dataset but not the ged dataset
#         # the number of differences are used to adjust the indexs of the start and end of each essay for indexind of the ged dataset.
#         # this method was not yetd eveloped to remove characters that appear in the grammar dataset but not the essay dataset.
#         self.find_differences()
#         rows_from_ged = [self.ged.loc[(self.ged['end_word_ind']>m[0])& (self.ged['end_word_ind']<=m[1])] for m in self.new_indexes]
#         # get the words from the ged dataset (with whitespace) to form each essay
#         essays_from_ged = [' '.join(tmp_df['word']) for tmp_df in rows_from_ged]
#         # get the labels from the ged dataset corresponding to each essay
#         labels_from_ged = [tmp_df['labels'] for tmp_df in rows_from_ged]
#         return pd.concat([pd.DataFrame({'text':essays_from_ged,'labels':labels_from_ged}),self.essays[['answer-s','script-s','id']]],axis=1)

#     def find_differences(self):
#         # store list of text for essays with differences removed
#         self.new_text = []
#         # store new indexes for essays with differences removed
#         self.new_indexes = []
#         # store list of indexes of differences
#         self.new_indexes_errors = []
#         # iterate through the aes dataset with itertuples
#         for i,row in enumerate(self.essays.itertuples()):
#             # get essay with no whitespace and essay length (characters) from the aes dataset
#             aes_essay_no_ws , char_len = row[self.essay_col_index['text_no_ws']] , row[self.essay_col_index['essay_char_len']] 
#             # get start and end of essay as it appears in the aes set 
#             # but with the number of differences that have appeared so far between the two datasets subtracted from the start and end index
#             start_ind , end_ind = row[self.essay_col_index['start_ind']]-self.errors_count,row[self.essay_col_index['end_word_ind']]-self.errors_count
#             # use the start and end index to get the potentially matching characters between the ged and aes datasets 
#             ged_essay = self.all_words_no_ws[start_ind:end_ind]
#             # method to locate errors between the two datasets
#             self.locate_chars_in_aes_but_not_ged(aes_essay_no_ws,ged_essay,char_len,start_ind,end_ind)

#     def locate_chars_in_aes_but_not_ged(self,aes_essay_no_ws,ged_essay,char_len,start_ind,end_ind):
#         # if the two essays are equal append the indexes to the new_index list
#         if aes_essay_no_ws==ged_essay:
#             self.new_indexes.append([start_ind,end_ind])
#         else:
#             # counter for number of characters in aes essay but not the ged dataset
#             aes_essay_errors_count = 0
#             # find the first character in aes essay but not the ged dataset
#             current_error = min(np.nonzero(np.invert(np.array(list(aes_essay_no_ws))==np.array(list(ged_essay))))[0])
#             # get the text which is in the ged essay up to the first difference 
#             correct_text = ged_essay[:current_error]
#             more_errors = True
#             while more_errors==True:
#                 # get the characters in the the aes essay beyond the current character which is found to be a difference between the two datasets
#                 tmp_aes_essay = aes_essay_no_ws[current_error+aes_essay_errors_count+1:char_len]
#                 # get the characters beyond the current difference in the ged essay
#                 tmp_ged = ged_essay[current_error:char_len-aes_essay_errors_count-1]
#                 if tmp_aes_essay == tmp_ged:
#                     # update the corrected text to contain the ged essay text before the current error
#                     correct_text = correct_text + tmp_ged
#                     # add the number of errors to the errors count
#                     aes_essay_errors_count = aes_essay_errors_count if aes_essay_errors_count!= 0 else 1
#                     self.errors_count+=aes_essay_errors_count
#                     more_errors = False
#                     break
#                 else:
#                     # find the number of characters between the current difference and next one
#                     char_to_next_error = min(np.nonzero(np.invert(np.array(list(tmp_aes_essay))==np.array(list(tmp_ged))))[0])
#                     # get the index of the next difference in the essay 
#                     current_error += char_to_next_error
#                     # update the corrected text to contain the ged essay text before the current error
#                     correct_text = correct_text + tmp_ged[:char_to_next_error]
#                     # add an error to the current essay difference count
#                     aes_essay_errors_count+=1
    ###################################################################################################################################