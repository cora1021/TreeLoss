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
import gensim.models as gsm


# emoticon = {'😂', '😷', '😮', '🙋', '🙄', '😐', '🙃', '😇', '😖', '😥', '😑', '😣', '😩', '😛', '😯', '🙁', '😱', '😕', '🙉', '😴', '😵', '😲', '😫', '😈', '😌', '😤', '🙂', '😎', '😨', '😻', '😒', '😰', '😋', '🙈', '😶', '😓', '🙅', '😼', '😧', '😪', '😟', '😘', '🙊', '😡', '😔', '🙀', '🙇', '😠', '🙍', '😬', '😾', '😳', '🙆', '😗', '😽', '😸', '😚', '🙏', '😞', '🙌', '😏', '😉', '😅', '😜', '😄', '😙', '😝', '😁', '😍', '🙎', '😦', '😊', '😀', '😺', '😢', '😿', '😭', '😆', '😃', '😹'}
emojis = list(emoji.EMOJI_UNICODE.values())
emoticon = []
e2v = gsm.KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True)
num = 0
for emoji in emojis:
    if emoji in e2v:
        num+=1
        emoticon.append(emoji)

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

    def create_input(self, input_strings, max_seq_length):

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
        return torch.LongTensor(input_ids_all), torch.FloatTensor(input_mask_all)
    



def get_labels(emoji_list):
  
  label2index = dict()
  for idx, element in enumerate(emoji_list):
    label2index[element] = idx

  return label2index

# def get_data(line):
#     f = open(file_name,'r',encoding='utf8')
#     X = []
#     Y = []
#     lang = []
#     for line in f:
#         line = json.loads(line)
#         text = line['text']
#         emoji = line['emoji_id']
#         language = line['language']
#         X.append(text)
#         Y.append(emoji)
#         lang.append(language)
#     return X, Y, lang


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
    
    for char in text:
        if char in emoticon:
            return True
    return False

# def extract_user_id(line):
    
#     if 'text' in line and 'user' in line and 'id' in line['user']:               
#         text = line['text']
#         if if_include_emoticon(text):
#             id_num = line['user']['id']
#         else:
#             id_num = None
                
#     return id_num

def data_process(tweet, label2index):
    
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
    X = []
    Y = []
    country = []
    languages = []
    user = []

    if id_num == None:
        dataline = list(zip(user,X,Y,country,languages))
        return dataline

    filted_char = []
    for char in tweet['text']:
        if char in emoticon: ## target_emoji
            target_labels.add(label2index[char])
        else:
            filted_char.append(char)

    text = ''.join(filted_char)
    tokens = text.split(' ')
    
    for token in tokens:
        
        # if token in UNICODE_EMOJI: ## other_emoji            
        #     continue
        if token.startswith('@'): ## user mention
            filted_tokens.append('<mention>')
            
        elif token.startswith('http'):  ## url
            filted_tokens.append('<url>')
            
        else:  #text
            filted_tokens.append(token)
    sentence = ' '.join(filted_tokens)
    
    for label in target_labels:
        user.append(id_num)
        X.append(sentence)
        Y.append(label)
        country.append(country_code)
        languages.append(language)
    dataline = list(zip(user,X,Y,country,languages))
    return dataline
