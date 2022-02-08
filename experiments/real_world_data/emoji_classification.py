import argparse
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--loss', choices=['xentropy', 'simloss', 'covertreeloss'], required=True)
parser.add_argument('--batch_size',type=int,default=8)
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--lower_bound', type=float, default=0.5)

args = parser.parse_args()

import numpy as np
import random
import os
import json
import sys
import torch
import torch.nn as nn
import time
from tool import BERTSentenceEncoder, data_process
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import logging
from sklearn.metrics import f1_score
import zipfile
from TreeLoss.loss import CoverTreeLoss
from torch.utils.data import IterableDataset
import gensim.models as gsm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock, chebyshev,cosine
from collections import defaultdict
from TreeLoss.utilities import set_logger, set_seed, _print, path, gen_sim

set_seed(args.seed)

writer = SummaryWriter(f'logging/emoji/{args.loss}_{args.lower_bound}_1974')
import emoji
emojis = list(emoji.EMOJI_UNICODE.values())
descrip = list(emoji.EMOJI_UNICODE.keys())

class Accuracy:
    def __init__(self):

        self.correct = 0
        self.count = 1e-10

    def __call__(self, pred, labels, sim):

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
    def __init__(self, length, sentence_encoder, num_labels, new2index):
        super(MyModel, self).__init__()
        self.bert = sentence_encoder
        self.classifier = nn.Linear(768, length)
        self.classifier_ = nn.Linear(768, num_labels)
        self.soft_max = nn.Softmax(dim=-1)
        self.metric = Accuracy()
        self.true_class_num = num_labels
        self.covertreeloss = CoverTreeLoss(num_labels, length, 768, new2index)
        self.simloss = SimLoss(w=sim_matrix.cuda())
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, tokens, mask, labels):
        output = self.bert(tokens, mask)
        if args.loss == 'xentropy':
            logits = self.classifier_(output)
            loss = self.cross_entropy(logits, labels)
            prob = self.soft_max(logits)
        if args.loss == 'simloss':
            logits = self.classifier_(output)
            prob = self.soft_max(logits)
            loss = self.simloss(prob,labels)
        if args.loss == 'covertreeloss':
            weights = self.classifier.weight
            loss, logits, added_weights = self.covertreeloss(weights, output, labels)
            prob = self.soft_max(logits)
        _, pred = prob.max(-1)
        
        return loss, prob, pred

    def predict(self, tokens, mask):

        output_layer = self.bert(tokens, mask)
        logits = self.classifier(output_layer)
        prob = self.soft_max(logits[:, :self.true_class_num])
        _, pred = prob.max(-1)

        return prob, pred

class Framework:
    def __init__(self, length, batch_size, num_labels):

        self.max_seq_length = 64
        self.pretrain_path = "./labse_bert_model"
        self.lexical_dropout = 0.3
        self.lower = True
        self.length = length
        self.num_labels = num_labels
        self.sentence_encoder = BERTSentenceEncoder(self.pretrain_path, self.lexical_dropout, self.lower)
        self.model = MyModel(self.length, self.sentence_encoder, self.num_labels, new2index)
        self.path = f'/data/yujie.wang/emoji_model_{args.loss}_{args.lower_bound}.zip' 
        self.initial = './bert_last_layer.zip'
        # self.model.load_state_dict(torch.load(self.path))

        if torch.cuda.is_available():
            self.model.cuda()

        timestamp = str(time.time())
        self.emoji_covertree_logger = set_logger(f'emoji_{args.loss}_{args.lower_bound}', timestamp)
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
        self.optimizer = AdamW(parameters_to_optimize, lr=2e-6, correct_bias=False)
        
        self.batch_size = batch_size


    def train(self, train_X, train_Y, it, sim_matrix):
        
        self.model.train()
        y_pred = []
        y_true = []
        batch = it
        for start, end in zip(range(0, len(train_X), self.batch_size),range(self.batch_size, len(train_X), self.batch_size)):
            batch += 1
            train_inputs = train_X[start:end]
            train_labels = train_Y[start:end]
            labels = torch.LongTensor(train_labels)
            tokens, masks = self.sentence_encoder.create_input(train_inputs, self.max_seq_length)
            if torch.cuda.is_available():
                tokens = tokens.cuda()
                masks = masks.cuda()
                labels = labels.cuda()
            

            loss, prob, pred = self.model(tokens, masks, labels)
            y_true+=labels.detach().cpu().tolist()
            y_pred+=pred.detach().cpu().tolist()
            
            loss.backward()
            # grad_norm = sum([ torch.norm(p.grad)**2 for p in self.model.parameters() if p.grad is not None])**(1/2)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.model.metric(pred, labels, sim_matrix)

        accuracy = self.model.metric.get_metric()
        F1_macro = f1_score(y_true, y_pred, average='macro')
        self.emoji_covertree_logger.info('Accuracy: \t {0:2.4f}  F1_macro: \t {1:2.4f} \n'.format(accuracy, F1_macro))
        # # sys.stdout.write('Accuracy: \t {0:2.4f}  F1_macro: \t {1:2.4f} Loss: \t {2:2.4f} \n'.format(accuracy, F1_macro, loss))
        # sys.stdout.flush()
        writer.add_scalar('Accuarcy on training', accuracy, it)
        writer.add_scalar('F1_macro on training', F1_macro, it)

        self.model.metric.reset()
        

    

    def dev(self, inputs, devset_labels, itt, sim_matrix, F1_best, accuracy_best):

        self.model.eval()
        
        dev_ypred = []
        dev_true = []
        for start, end in zip(range(0, len(inputs), self.batch_size),range(self.batch_size, len(inputs), self.batch_size)):
            dev_inputs = inputs[start:end]
            dev_labels = devset_labels[start:end]
            labels = torch.LongTensor(dev_labels)
            tokens, masks = self.sentence_encoder.create_input(dev_inputs, self.max_seq_length)
            if torch.cuda.is_available():
                tokens = tokens.cuda()
                masks = masks.cuda()
                labels = labels.cuda()
            dev_loss, dev_prob, dev_pred = self.model(tokens, masks, labels)
            self.model.metric(dev_pred, labels, sim_matrix)

            dev_true+=labels.detach().cpu().tolist()
            dev_ypred+=dev_pred.detach().cpu().tolist()

        accuracy = self.model.metric.get_metric()
        F1_macro = f1_score(dev_true, dev_ypred, average='macro')
    
        self.emoji_covertree_logger.info('Accuracy on dev: \t {0:2.4f}  F1_macro on dev: \t {1:2.4f} \n'.format(accuracy, F1_macro))
        writer.add_scalar('Accuracy on dev', accuracy,itt)
        writer.add_scalar('F1 score on dev', F1_macro,itt)
        if accuracy>accuracy_best:
            torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.path)

        return accuracy, F1_macro


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
        
        # return F1_macor, f1

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

emoji_vector = []
use_emoji = []
e2v = gsm.KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True)
for emoji in emojis:
    if emoji in e2v:
        use_emoji.append(emoji)
        vector = e2v[emoji]
        emoji_vector.append(vector)
label2index = dict()
for idx, element in enumerate(use_emoji):
    label2index[element] = idx
num_labels = len(label2index)
emoji_matrix = np.array(emoji_vector)
# simloss
sim_matrix = gen_sim(emoji_matrix)
sim = np.copy(sim_matrix)
sim_matrix = (sim_matrix - args.lower_bound) / (1 - args.lower_bound)
sim_matrix[sim_matrix < 0.0] = 0.0
# covertree loss
new2index, length = CoverTreeLoss.tree_structure(num_labels,emoji_matrix)
framework = Framework(length, args.batch_size, num_labels)
train_X = []
train_Y = []

dev_X = []
dev_Y =[]

test_X = []
test_Y = []

large_prime_value = 9999
it = 0
itt = 0
F1_best = 0
accuracy_best = 0
dataloader = Dataset()

for dataline in dataloader:      
    for line in dataline:
        
        user_id,X,Y,country,lang = line

        random_value = user_id*large_prime_value%100
        if random_value<80:         
            train_X.append(X)
            train_Y.append(Y)
            # print(len(train_X))

        elif random_value<90:         
            dev_X.append(X)
            dev_Y.append(Y)
        else:
            test_X.append(X)
            test_Y.append(Y)
    
        if len(train_X) >= 8000:
            
            train_data = list(zip(train_X, train_Y))
            random.shuffle(train_data)
            train_X, train_Y = zip(*train_data)
            framework.train(train_X,train_Y, it,sim)
            train_X = []
            train_Y = []
      
            it += 1

        if len(dev_X)>=8000:
            dev_accuracy, F1_current = framework.dev(dev_X, dev_Y, itt, sim, F1_best, accuracy_best)
            if F1_current > F1_best:
                F1_best = F1_current
            if dev_accuracy > accuracy_best:
                accuracy_best = dev_accuracy
            dev_X = []
            dev_Y = []
            itt += 1