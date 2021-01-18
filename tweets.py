import numpy as np
import random
import os
import json
import sys
import torch
import torch.nn as nn
import time
from tool import get_labels, get_data, BERTSentenceEncoder,split_y,if_include_emoticon,extract_user_id,data_process
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import logging
from sklearn.metrics import f1_score
import zipfile
from loss import SimLoss
from torch.utils.data import IterableDataset
import gensim.models as gsm
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--batch_size',type=int,default=8)

args = parser.parse_args()

writer = SummaryWriter('runs/CCE_experiment')
import emoji
# emoticon = ['😂', '😷', '😮', '🙋', '🙄', '😐', '🙃', '😇', '😖', '😥', '😑', '😣', '😩', '😛', '😯', '🙁', '😱', '😕', '🙉', '😴', '😵', '😲', '😫', '😈', '😌', '😤', '🙂', '😎', '😨', '😻', '😒', '😰', '😋', '🙈', '😶', '😓', '🙅', '😼', '😧', '😪', '😟', '😘', '🙊', '😡', '😔', '🙀', '🙇', '😠', '🙍', '😬', '😾', '😳', '🙆', '😗', '😽', '😸', '😚', '🙏', '😞', '🙌', '😏', '😉', '😅', '😜', '😄', '😙', '😝', '😁', '😍', '🙎', '😦', '😊', '😀', '😺', '😢', '😿', '😭', '😆', '😃', '😹']
emojis = list(emoji.EMOJI_UNICODE.values())
descrip = list(emoji.EMOJI_UNICODE.keys())
label2index = get_labels(emojis)
num_labels = len(label2index)

class Accuracy:
    def __init__(self):

        self.correct = 0
        self.count = 1e-10

    def __call__(self, pred, labels):

        _pred = pred.detach().cpu()
        _labels = labels.detach().cpu()

        self.correct+=torch.eq(_pred, _labels).sum().item()
        self.count += _labels.size(0)

    def get_metric(self):

        return self.correct/self.count

    def reset(self):

        self.correct = 0
        self.count = 1e-10

class MyModel(nn.Module):
    def __init__(self, num_labels, sentence_encoder):
        super(MyModel, self).__init__()
        self.bert = sentence_encoder
        self.classifier = nn.Linear(768, num_labels)
        self.soft_max = nn.Softmax(dim=-1)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.metric = Accuracy()
        # self.simloss = SimCE(w=sim_matrix)
    
    def forward(self, tokens, mask, labels):
        output_layer = self.bert(tokens, mask)
        logits = self.classifier(output_layer)
        loss = self.cross_entropy(logits, labels)
        prob = self.soft_max(logits)
        # loss = self.simloss(prob,labels)  #Simloss function
        _, pred = prob.max(-1)
        
        return loss, prob, pred

    def predict(self, tokens, mask):

        output_layer = self.bert(tokens, mask)
        logits = self.classifier(output_layer)
        prob = self.soft_max(logits)
        _, pred = prob.max(-1)

        return prob, pred




def set_logger(name, timestamp):
    formatter=logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger=logging.getLogger(f'{name}-Logger')
    file_handler=logging.FileHandler(f'./log-{name}-{timestamp}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler=logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger

class Framework:
    def __init__(self, batch_size, num_labels):

        self.max_seq_length = 64
        self.pretrain_path = "./labse_bert_model"
        self.lexical_dropout = 0.3
        self.lower = True
        self.num_labels = num_labels
        self.sentence_encoder = BERTSentenceEncoder(self.pretrain_path, self.lexical_dropout, self.lower)
        self.model = MyModel(self.num_labels, self.sentence_encoder)
        self.path = './last_layer.zip' # save last layer model
        self.model.load_state_dict(torch.load(self.path))

        if torch.cuda.is_available():
            self.model.cuda()

        timestamp = str(time.time())
        self.per_batch_logger = set_logger('per_batch', timestamp)
        self.per_1000_batch_logger = set_logger('per_1000_batch', timestamp)
        parameters_to_optimize = []

        ## train last layer
        for n, p in list(self.model.named_parameters()):
            if p.requires_grad:
                parameters_to_optimize.append((n, p))

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        self.optimizer = AdamW(parameters_to_optimize, lr=2e-5, correct_bias=False)
        
        self.batch_size = batch_size


    def train(self, train_X, train_Y,train_country,train_lang,it):
        
        self.model.train()
        y_pred = []
        y_true = []
        langs = defaultdict(lambda: {})
        langs = {
            'true': defaultdict(lambda: []),
            'pred': defaultdict(lambda: []),
            }
        countries =defaultdict(lambda: [])
        countries ={
            'true': defaultdict(lambda: []),
            'pred': defaultdict(lambda: []),
        }
        # label_class =defaultdict(lambda: [])
        # label_class ={
        #     'true': defaultdict(lambda: []),
        #     'pred': defaultdict(lambda: []),
        # }
        batch = 1000*it
        for start, end in zip(range(0, len(train_X), self.batch_size),range(self.batch_size, len(train_X), self.batch_size)):
            batch += 1
            train_inputs = train_X[start:end]
            train_labels = train_Y[start:end]
            
            tokens, masks, labels = self.sentence_encoder.create_input(train_inputs, self.max_seq_length, train_labels)
            if torch.cuda.is_available():
                tokens = tokens.cuda()
                masks = masks.cuda()
                labels = labels.cuda()
            

            loss, prob, pred = self.model(tokens, masks, labels)
            y_true+=labels.detach().cpu().tolist()
            y_pred+=pred.detach().cpu().tolist()

            loss.backward()
            grad_norm = sum([ torch.norm(p.grad)**2 for p in self.model.parameters() if p.grad is not None])**(1/2)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            writer.add_scalar('Norm_grad:  ', grad_norm.item(), batch)
            writer.add_scalar('Loss:  ', loss.item(), batch)
            # self.per_batch_logger.info('Grad_norm: \t {0:2.4f}  Loss: \t {1:2.4f} \n'.format(grad_norm,loss))
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.model.metric(pred, labels)
            
        #evaluation
        for xs in range(len(y_true)):
            langs['true'][train_lang[xs]].append(y_true[xs])
            langs['pred'][train_lang[xs]].append(y_pred[xs])
            countries['true'][train_country[xs]].append(y_true[xs])
            countries['pred'][train_country[xs]].append(y_pred[xs])
            # label_class['true'][y_true[xs]].append(y_true[xs])
            # label_class['pred'][y_true[xs]].append(y_pred[xs])

        accuracy = self.model.metric.get_metric()
        F1_macro = f1_score(y_true, y_pred, average='macro')
        F1_english = f1_score(langs['true']['en'],langs['pred']['en'],average='macro')
        F1_US = f1_score(countries['true']['US'],countries['pred']['US'],average='macro')
        
        self.per_1000_batch_logger.info('Accuracy: \t {0:2.4f}  F1_macro: \t {1:2.4f} F1_english: \t {2:2.4f} F1_US: \t {3:2.4f} \n'.format(accuracy, F1_macro,F1_english,F1_US))
        # sys.stdout.write('Accuracy: \t {0:2.4f}  F1_macro: \t {1:2.4f} Loss: \t {2:2.4f} \n'.format(accuracy, F1_macro, loss))
        sys.stdout.flush()
        writer.add_scalar('Accuarcy: ', accuracy, it)
        writer.add_scalar('F1_macro:  ', F1_macro, it)
        writer.add_scalar('F1_english:  ', F1_english, it)
        writer.add_scalar('F1_US:  ', F1_english, it)
        # writer.add_scalar('loss_class0:  ', loss_0, it)


        
        self.model.metric.reset()
        

    

    def dev(self, inputs, devset_labels, F1_best,itt):

        self.model.eval()
        
        dev_ypred = []
        dev_true = []
        for start, end in zip(range(0, len(inputs), self.batch_size),range(self.batch_size, len(inputs), self.batch_size)):
            dev_inputs = inputs[start:end]
            dev_labels = devset_labels[start:end]
            
            tokens, masks, labels = self.sentence_encoder.create_input(dev_inputs, self.max_seq_length, dev_labels)
            if torch.cuda.is_available():
                tokens = tokens.cuda()
                masks = masks.cuda()
                labels = labels.cuda()
            dev_prob, dev_pred = self.model.predict(tokens, masks)
            # self.model.metric(dev_pred, labels)

            dev_true+=labels.detach().cpu().tolist()
            dev_ypred+=dev_pred.detach().cpu().tolist()

        # accuracy = self.model.metric.get_metric()
        F1_macro = f1_score(dev_true, dev_ypred, average='macro')
        self.per_1000_batch_logger.info(f'F1 score on dev {F1_macro}')
        writer.add_scalar('F1 score on dev', F1_macro,itt)
        if F1_macro>F1_best:
            torch.save(self.model.state_dict(), self.path)

        return F1_macro


    def test(self):
        sentence_encoder = BERTSentenceEncoder(self.pretrain_path, self.lexical_dropout, self.lower)
        model = MyModel(self.num_labels, sentence_encoder)
        model.load_state_dict(torch.load(self.path))
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        test_true = []
        test_ypred = []
        for start, end in zip(range(0, self.num_test, self.batch_size),range(self.batch_size, self.num_test, self.batch_size)):
            test_inputs = self.testX[start:end]
            test_labels = self.testY[start:end]

            tokens, masks, labels = self.sentence_encoder.create_input(test_inputs, self.max_seq_length, test_labels)
            if torch.cuda.is_available():
                tokens = tokens.cuda()
                masks = masks.cuda()
                labels = labels.cuda()
            test_prob, test_pred = model.predict(tokens, masks)
            
            # model.metric(test_pred, labels)
            test_true+=test_labels.detach().cpu().tolist()
            test_ypred+=test_pred.detach().cpu().tolist()

        F1_macro = f1_score(test_true, test_ypred, average='macro')
        
        return F1_macor, f1

class Dataset(IterableDataset):
    def __iter__(self):
        return Dataset.scramble(Dataset.yield_data(), 809600)
    
    def yield_data():
        '''
        creates a generator that yields a single tweet at a time;
        the results are yielded in the order they are present on disk,
        and so the results must be combined with the scramble function 
        to generate a random sample
        '''
        file_path = '/data/Twitter dataset'
        zip_name = [file_path+'/'+name for name in os.listdir(file_path) if name.endswith('.zip')]
        random.shuffle(zip_name)
        for name in zip_name:
            file = zipfile.ZipFile(name,'r')
            file_name = file.namelist()
            random.shuffle(file_name)
            for names in file_name:
                f = file.open(names,'r')
                
                for line in f:
                    try:
                        line = json.loads(line)        
                        dataline = data_process(line,label2index)
                    except:
                        continue
                        
                    if len(dataline)>0:
                        yield dataline

    def scramble(gen, buffer_size):
        '''
        randomizes the order of a generator using O(buffer_size) memory instead of O(n);
        the downside is that the results are not truly uniformly random,
        but they are random enough for training purposes
        '''
        buf = []
        i = iter(gen)
        while True:
            try:
                e = next(i)
                buf.append(e)
                if len(buf) >= buffer_size:
                    choice = random.randint(0, len(buf)-1)
                    buf[-1],buf[choice] = buf[choice],buf[-1]
                    yield buf.pop()
            except StopIteration:
                random.shuffle(buf)
                yield from buf
                return

#simloss
# emoticon_vector = []
# e2v = gsm.KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True)
# for emoji in emoticon:
#     vector = e2v[emoji]
#     emoticon_vector.append(vector)


# cosine_dist = cosine_similarity(emoticon_vector)
# lower_bound = 0.5
# sim_matrix = torch.from_numpy(cosine_dist)
# sim_matrix = (sim_matrix - lower_bound) / (1 - lower_bound)
# sim_matrix[sim_matrix < 0.0] = 0.0
# reduction_factor = 0

framework = Framework(args.batch_size, num_labels)
train_X = []
train_Y = []
train_country = []
train_lang = []
dev_X = []
dev_Y =[]
dev_country = []
dev_lang = []
test_X = []
test_Y = []
test_country = []
test_lang = []
large_prime_value = 9999
it = 0
itt = 0
F1_best = 0
dataloader = Dataset()

for dataline in dataloader:      
    for line in dataline:
        
        user_id,X,Y,country,lang = line
        random_value = user_id*large_prime_value%100
        if random_value<80:          
            train_X.append(X)
            train_Y.append(Y)
            train_country.append(country)
            train_lang.append(lang)
        elif random_value<90:         
            dev_X.append(X)
            dev_Y.append(Y)
            dev_country.append(country)
            dev_lang.append(lang)
        else:
            test_X.append(X)
            test_Y.append(Y)
            test_country.append(country)
            test_lang.append(lang)
    
        if len(train_X) >= 8000:
            train_data = list(zip(train_X, train_Y,train_country,train_lang))
            random.shuffle(train_data)
            train_X, train_Y, train_country, train_lang = zip(*train_data)
            framework.train(train_X,train_Y,train_country,train_lang,it)
            train_X = []
            train_Y = []
            train_country = []
            train_lang = []
            it += 1

        # if len(dev_X)>=10000:
        #     F1_current = framework.dev(dev_X,dev_Y,F1_best,itt)
        #     if F1_current > F1_best:
        #         F1_best = F1_current
        #     dev_X = []
        #     dev_Y = []
        #     dev_country = []
        #     dev_lang = []
        #     itt += 1