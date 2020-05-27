# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# %%
# Files
import input_net
import utils

# Packages
from os import path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime

# Transformers
from transformers import AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer
from transformers import BartConfig, BartForSequenceClassification, BartTokenizer
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import CamembertConfig, CamembertForSequenceClassification, CamembertTokenizer
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from transformers import XLMConfig, XLMForSequenceClassification, XLMTokenizer
from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from transformers import XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer
from sklearn.model_selection import train_test_split

from transformers import AdamW, get_linear_schedule_with_warmup

# %% [markdown]
# ## Models

# %%
#                       Model                                   Tokenizer               Pretrained weights shortcut 
MODEL_CLASSES = {
    'albert': (         AlbertForSequenceClassification,        AlbertTokenizer,        'albert-large-v2'),
    'bart': (           BartForSequenceClassification,          BartTokenizer,          'bart-large'),
    'bert': (           BertForSequenceClassification,          BertTokenizer,          'bert-base-uncased'),
    'camembert': (      CamembertForSequenceClassification,     CamembertTokenizer,     'camembert-base'),
    'distilbert': (     DistilBertForSequenceClassification,    DistilBertTokenizer,    'distilbert-base-uncased'),
    'flaubert': (       FlaubertForSequenceClassification,      FlaubertTokenizer,      'flaubert-base-uncased'),
    'roberta': (        RobertaForSequenceClassification,       RobertaTokenizer,       'roberta-base'),
    'xlm': (            XLMForSequenceClassification,           XLMTokenizer,           'xlm-mlm-en-2048'),
    'xlm_roberta':(     XLMRobertaForSequenceClassification,    XLMRobertaTokenizer,    'xlm-roberta-base'),
    'xlnet': (          XLNetForSequenceClassification,         XLNetTokenizer,         'xlnet-base-cased')
}

args = {
    'model_type': 'bert',
    'do_train': True,
    'do_eval': True,
    'max_seq_length': 70,
    'batch_size': 8, 
    'epochs': 2,
    'learning_rate': 1e-3,
    'num_training_steps': 100,
    'num_warmup_steps': 100,
    'max_grad_norm': 1.0
}

model_class, tokenizer_class, pretrained_model = MODEL_CLASSES[args['model_type']]

# %% [markdown]
# ## Input Generation

# %%
TRAIN = "data/train.csv"
TEST = "data/test.csv"
INPUT_NET = 'data/input' + str(args['max_seq_length']) + '_' + pretrained_model + '.csv'

if not path.exists(INPUT_NET):
    df = input_net.create_input(TRAIN, INPUT_NET, tokenizer_class, pretrained_model, args)
else:
    df = pd.read_csv(INPUT_NET)

# %% [markdown]
# ## Net Functions -> quan funcioni tot cridar-ho desde net.py
# 

# %%
def batch_generator(data, target, batch_size):
    data = np.array(data)[:100]
    target = np.array(target)[:100]
    nsamples = len(data)
    perm = np.random.permutation(nsamples)
    for i in range(0, nsamples, batch_size):
        batch_idx = perm[i:i+batch_size]
        if (i//batch_size % 5 == 0 or i//batch_size == 0):
            print("Batch", i//batch_size+1 , "of", nsamples//batch_size+1)
        if target is not None:
            yield data[batch_idx,:], target[batch_idx]
        else:
            yield data[batch_idx], None

def training(model, train_data, train_target, epoch, args):

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = AdamW(model.parameters(), lr=args['learning_rate'], correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args['num_warmup_steps'],
                                                num_training_steps=args['num_training_steps'])
    step = 0
    ncorrect = 0
    total_loss = 0

    batch_size = args['batch_size']
    model.train()

    for X, y in batch_generator(train_data, train_target, batch_size):

        X_i, X_s, X_p, y = utils.ToTensor(X,y)
        
        model.zero_grad()

        out = model(input_ids=X_i, token_type_ids=X_s, attention_mask=X_p, labels=y)[1]
        
        loss = criterion(out, y)
        loss.backward()
        total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])  # Gradient clipping 
        optimizer.step()
        scheduler.step()

        out = F.softmax(out, dim=1)

        ncorrect += (torch.max(out, 1)[1] == y).sum().item()
        
        if (step % 5 == 0 or step == 0):
            print("Training Loss : {0:.2f}".format(loss))
        step += 1

    total_loss /= len(train_data)
    acc = ncorrect/len(train_data) * 100
    print("\nAverage Training Accuracy: {0:.2f} \n".format(acc))
    return acc, loss


def validation(model, eval_data, eval_target, epoch, args):
    criterion = nn.CrossEntropyLoss(reduction='mean')

    step = 0
    ncorrect = 0
    total_loss = 0

    batch_size = args['batch_size']
    model.eval()
    for X, y in batch_generator(eval_data, eval_target, batch_size):
        
        X_i, X_s, X_p, y = utils.ToTensor(X,y)
        
        out = model(input_ids=X_i, token_type_ids=X_s, attention_mask=X_p, labels=y)[1]
        
        loss = criterion(out, y)   
        total_loss += loss
        out = F.softmax(out, dim=1)
        ncorrect += (torch.max(out, 1)[1] == y).sum().item()
        if (step % 5 == 0 or step == 0):
            print("Validation Loss : {0:.2f}".format(loss))
        step += 1

    total_loss /= len(eval_data)
    acc = ncorrect/len(eval_data) * 100
    print("\nAverage Validation Accuracy: {0:.2f} \n".format(acc))
    return acc, loss


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def build(learn_data, model_class, pretrained_model, args):

    model = model_class.from_pretrained(pretrained_model, num_labels=2)
    print("Model loaded")

    epochs = args['epochs']

    X_train, X_val, y_train, y_val = train_test_split(learn_data.iloc[:,:-1], learn_data.iloc[:,-1],                                                            test_size=0.2, random_state=47)

    train_acc = [None]*epochs
    train_loss = [None]*epochs
    val_acc = [None]*epochs
    val_loss = [None]*epochs

    total_t0 = time.time()

    for epoch in range(0, epochs):
        print("")
        print('EPOCH {:} / {:} '.format(epoch + 1, epochs))
        print('======== Training ========')
        total_train_loss = 0

        t0 = time.time()
        t_acc, t_loss = training(model, X_train, y_train, epoch, args)
             
        train_acc[epoch] = t_acc
        train_loss[epoch] = t_loss
        
        print('======== Validació ========')
        v_acc, v_loss = validation(model, X_val, y_val, epoch, args)
        val_acc[epoch] = v_acc
        val_loss[epoch] = v_loss

    model.save_pretrained('trained/models')  

    return train_acc, val_acc

def test(model, test_data, args):
    ncorrect = 0
    total_loss = 0

    batch_size = args['batch_size']

    for X, _ in batch_generator(test_data, batch_size):
        X_i, X_s, X_p = utils.ToTensor(X,y=None)
        
        out = model(input_ids=X_i, token_type_ids=X_s, attention_mask=X_p)[0]
        out = F.softmax(out, dim=1)

        ncorrect += (torch.max(out, 1)[1] == y).sum().item()

    acc = ncorrect/len(test_data) * 100
    return acc

# %% [markdown]
# ## Learning

# %%

train_acc, val_acc = build(df, model_class, pretrained_model, args)
print("-------------------------------")
print("Final Training Accuracy {0:.2f}".format(train_acc[-1]))
print("Final Validation Accuracy {0:.2f}".format(val_acc[-1]))
print("-------------------------------")
# %% [markdown]
# ## Proves

# %%
#model = model_class.from_pretrained(pretrained_model, num_labels=2)
#X = np.array(df.iloc[:,:-1])
#y = np.array(df.iloc[:,-1])
#X_i, X_s, X_p, y = utils.ToTensor(X,y)

# Predict
#Predictions = model(input_ids=X_i[:5],token_type_ids=X_s[:5],attention_mask=X_p[:5])
# Training
#Predictions = model(input_ids=X_i[:5],token_type_ids=X_s[:5],attention_mask=X_p[:5], labels=y[:5])

