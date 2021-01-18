import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import os
from collections import defaultdict
import zipfile
# from emoji import UNICODE_EMOJI
import emoji

# emoticon = {'😂', '😷', '😮', '🙋', '🙄', '😐', '🙃', '😇', '😖', '😥', '😑', '😣', '😩', '😛', '😯', '🙁', '😱', '😕', '🙉', '😴', '😵', '😲', '😫', '😈', '😌', '😤', '🙂', '😎', '😨', '😻', '😒', '😰', '😋', '🙈', '😶', '😓', '🙅', '😼', '😧', '😪', '😟', '😘', '🙊', '😡', '😔', '🙀', '🙇', '😠', '🙍', '😬', '😾', '😳', '🙆', '😗', '😽', '😸', '😚', '🙏', '😞', '🙌', '😏', '😉', '😅', '😜', '😄', '😙', '😝', '😁', '😍', '🙎', '😦', '😊', '😀', '😺', '😢', '😿', '😭', '😆', '😃', '😹'}
emoticon = list(emoji.EMOJI_UNICODE.values())

class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, lexical_dropout, lower): 
        nn.Module.__init__(self)
        self.bert = AutoModel.from_pretrained(pretrain_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
        self.lower = lower

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x
        
    def forward(self, tokens, mask):
        
        last_layer, outputs = self.bert(tokens, attention_mask=mask)
        outputs = torch.mean(last_layer,dim=1)
        return outputs
    
    def tokenize(self, raw_tokens):
        # token -> index
           
        tokens = ['[CLS]']

        for i, token in enumerate(raw_tokens):
            
            if self.lower:
                input_token = token.lower()
            else:
                input_token = token

            sub_words = self.tokenizer.tokenize(input_token)
            tokens += sub_words

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        return indexed_tokens

    def create_input(self, input_strings, max_seq_length, labels):

        input_ids_all, input_mask_all = [], []
        for input_string in input_strings:
            # Tokenize input.
            input_tokens = self.tokenize(input_string.split(' '))
            sequence_length = min(len(input_tokens), max_seq_length)

            # Padding or truncation.
            if len(input_tokens) >= max_seq_length:
                input_tokens = input_tokens[:max_seq_length]
            else:
                input_tokens = input_tokens + [0] * (max_seq_length - len(input_tokens))

            input_mask = [1] * sequence_length + [0] * (max_seq_length - sequence_length)

            input_ids_all.append(input_tokens)
            input_mask_all.append(input_mask)
        return torch.LongTensor(input_ids_all), torch.FloatTensor(input_mask_all), torch.LongTensor(labels)
    



def get_labels(emoji_list):
  
  label2index = dict()
  for idx, element in enumerate(emoji_list):
    label2index[element] = idx

  return label2index

def get_data(line):
    f = open(file_name,'r',encoding='utf8')
    X = []
    Y = []
    lang = []
    for line in f:
        line = json.loads(line)
        text = line['text']
        emoji = line['emoji_id']
        language = line['language']
        X.append(text)
        Y.append(emoji)
        lang.append(language)
    return X, Y, lang


def create_input(input_strings, tokenizer, max_seq_length):

  input_ids_all, input_mask_all, segment_ids_all = [], [], []
  for input_string in input_strings:
    # Tokenize input.
    input_tokens = ["[CLS]"] + tokenizer.tokenize(input_string) + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    sequence_length = min(len(input_ids), max_seq_length)

    # Padding or truncation.
    if len(input_ids) >= max_seq_length:
      input_ids = input_ids[:max_seq_length]
    else:
      input_ids = input_ids + [0] * (max_seq_length - len(input_ids))

    input_mask = [1] * sequence_length + [0] * (max_seq_length - sequence_length)

    input_ids_all.append(input_ids)
    input_mask_all.append(input_mask)
    segment_ids_all.append([0] * max_seq_length)

  return np.array(input_ids_all), np.array(input_mask_all), np.array(segment_ids_all)


def split_y(y_pred,y_true,lang):
    lang_truth = defaultdict(list)
    for pred, y, l in zip(y_pred, y_true, lang):
        lang_truth[l].append([pred, y])
    
    for k, v in lang_truth.items():
        pred, y = [], []
        for line in v:
            p, truth = line
            pred.append(p)
            y.append(truth)
        lang_truth[k] = [pred, y]
    return lang_truth

def if_include_emoticon(text):
    tokens = text.split(' ')
    for token in tokens:
        if token in emoticon:
            return True
    return False

def extract_user_id(line):
    
    if 'text' in line and 'user' in line and 'id' in line['user']:               
        text = line['text']
        if if_include_emoticon(text):
            id_num = line['user']['id']
        else:
            id_num = None
                
    return id_num

def data_process(tweet, label2index):
    
    tokens = tweet['text'].split(' ')
    filted_tokens = []
    target_labels = set()
    country_code = tweet['place']['country_code']
    language = tweet['lang']

    if 'text' in tweet and 'user' in tweet and 'id' in tweet['user']:               
        text = tweet['text']
        if if_include_emoticon(text):
            id_num = tweet['user']['id']
        else:
            id_num = None

    for token in tokens:
        
        if token in emoticon: ## target_emoji
            target_labels.add(label2index[token])
            
        # elif token in UNICODE_EMOJI: ## other_emoji            
        #     continue
        elif token.startswith('@'): ## user mention
            filted_tokens.append('<mention>')
            
        elif token.startswith('http'):  ## url
            filted_tokens.append('<url>')
            
        else:  #text
            filted_tokens.append(token)
    sentence = ' '.join(filted_tokens)
    
    X = []
    Y = []
    country = []
    languages = []
    user = []
    if id_num:
        for label in target_labels:
            user.append(id_num)
            X.append(sentence)
            Y.append(label)
            country.append(country_code)
            languages.append(language)
    dataline = list(zip(user,X,Y,country,languages))
    return dataline
