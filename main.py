
# %%
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# %%

# If executed on Kaggle, set this parameter to True
kaggle = False

# %%
# Scripts files
if kaggle:
    from shutil import copyfile
    copyfile(src = "/kaggle/input/quorascripts/input_net.py", dst = "../working/input_net.py")
    copyfile(src = "/kaggle/input/quorascripts/utils.py", dst = "../working/utils.py")

import utils
import input_net

# Packages
from os import path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import datetime
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

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
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup


# %%
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

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
    'data_size': None, # If it is None: all data
    'model_type': 'bert',
    'seed_val' : 47,
    'max_seq_length': 70,
    'batch_size': 64,
    'epochs': 3,
    'learning_rate': 8e-6,
    'epsilon': 1e-8,
    'accum_steps': None,
    'met_sch': "constant",
    'warmup_proportion': 0.001,
    'max_grad_norm': 1.0,
    'do_train': True,
    'do_eval': True
}

model_class, tokenizer_class, pretrained_model = MODEL_CLASSES[args['model_type']]

# %% [markdown]
# ## Input Generation

# %%
if kaggle:
    TRAIN = "/kaggle/input/quora-data/train.csv"
    INPUT_NET = '/kaggle/input/quora-data/data/data/' + str(args['max_seq_length']) + "/input" + str(args['max_seq_length']) + '_' + pretrained_model + '.csv'
else:
    INPUT_NET = "data/" + str(args['max_seq_length']) + "/input" + str(args['max_seq_length']) + '_' + pretrained_model + '.csv'
    TRAIN = "data/train.csv"

if not path.exists(INPUT_NET):
    df = input_net.create_input(TRAIN, INPUT_NET, tokenizer_class, pretrained_model, args)
else:
    df = pd.read_csv(INPUT_NET)


# %%
# Data size
if args['data_size'] is not None:
    df = np.array(df)[:args['data_size']]
else:
    df = np.array(df)

X_i, X_s, X_p, y = utils.ToTensor(df[:,:-1],df[:,-1])

dataset = TensorDataset(X_i, X_s, X_p, y)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

random.seed(args['seed_val'])
np.random.seed(args['seed_val'])
torch.manual_seed(args['seed_val'])
torch.cuda.manual_seed_all(args['seed_val'])

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))


# %%
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = args['batch_size'] # Trains with this batch size.
        )

validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = args['batch_size'] # Evaluate with this batch size.
        )

print("Number of training batches:", len(train_dataloader))
print("Number of validation batches:", len(validation_dataloader))

model = model_class.from_pretrained(pretrained_model, num_labels=2, output_attentions = False, output_hidden_states = False).to(device)

if device == "cuda":
    model.cuda()
print("Model loaded")

criterion = nn.CrossEntropyLoss(reduction='mean') 

optimizer = AdamW(model.parameters(),
                  lr = args['learning_rate'],
                  eps = args['epsilon']
                )

num_steps = len(train_dataloader) * args['epochs'] 

if args['met_sch'] == "linear":
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = num_steps *  args['warmup_proportion'],
                                                num_training_steps = num_steps)

elif args['met_sch'] == "constant":
    scheduler = get_constant_schedule_with_warmup(optimizer,
                                              num_warmup_steps = num_steps *  args['warmup_proportion'])


# %%
def training():
    ncorrect = 0
    ntrue = 0
    npred = 0
    npredtrue = 0
    total_loss = 0
    model.train()
    start = time.time()
    
    for step, batch in enumerate(train_dataloader):
        if step % int(len(train_dataloader)/20) == 0 and not step == 0:
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))

        model.zero_grad()
        y = batch[3].to(device)
        
        if args['model_type'] == "distilbert" or args['model_type'] == "bart":
            out = model(input_ids=batch[0].to(device), attention_mask=batch[2].to(device), labels=y)[1]
        else:
            out = model(input_ids=batch[0].to(device), token_type_ids=batch[1].to(device), attention_mask=batch[2].to(device), labels=y)[1]

        loss = criterion(out, y)
        
        loss.backward()    
        torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
        optimizer.step()
        if args['met_sch']:
            scheduler.step()
        
        out = F.softmax(out, dim=1)
        pred = torch.max(out, 1)[1]
        
        total_loss += loss.item()*len(batch[0])
        ncorrect += (pred == y).sum().item()
        for i in range(out.shape[0]):
            if y[i]:
                ntrue += 1
            if pred[i]:
                npred += 1
                if y[i]:
                    npredtrue += 1
        
        
        print("Batch time: ", utils.format_time(end-start))
    end = time.time()
    avg_loss = total_loss / train_size
    acc = ncorrect/train_size * 100
    try:
        recall = npredtrue / ntrue
    except:
        recall = np.nan
    try:
        precision = npredtrue / npred
    except:
        precision = np.nan
    try:
        f1 = 2 / (1/recall + 1/precision)
    except:
        f1 = np.nan
    conf = [[train_size-(ntrue+npred-npredtrue), ntrue-npredtrue], [npred-npredtrue,npredtrue]]
    
    epoch_time = utils.format_time(end-start)
    
    print("")
    print("  epoch time:", epoch_time)
    print("  ncorrect:", ncorrect)
    print("  recall:", recall)
    print("  precision:", precision)
    print("  f1:", f1)
    print("  total_loss:", total_loss)

    print("\nAverage training accuracy: {0:.4f}".format(acc), "%")
    print("Average training loss: {0:.4f}".format(avg_loss))
    
    utils.confusion_matrix(conf)

    return acc, avg_loss

    
def validation():
    ncorrect = 0
    ntrue = 0
    npred = 0
    npredtrue = 0
    total_loss = 0
    model.eval()
    
    start = time.time()
    with torch.no_grad():
        for step, batch in enumerate(validation_dataloader):

            if step % int(len(validation_dataloader)/20) == 0 and not step == 0:
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(validation_dataloader)))
                
            y = batch[3].to(device)

            if args['model_type'] == "distilbert" or args['model_type'] == "bart":
                out = model(input_ids=batch[0].to(device), attention_mask=batch[2].to(device), labels=y)[1]
            else:
                out = model(input_ids=batch[0].to(device), token_type_ids=batch[1].to(device), attention_mask=batch[2].to(device), labels=y)[1]

            loss = criterion(out, y)
            out = F.softmax(out, dim=1)
            pred = torch.max(out, 1)[1]
            
            total_loss += loss.item()*len(batch[0])
            ncorrect += (pred == y).sum().item()
            for i in range(out.shape[0]):
                if y[i]:
                    ntrue += 1
                if pred[i]:
                    npred += 1
                    if y[i]:
                        npredtrue += 1
    
    end = time.time()
    avg_loss = total_loss / val_size
    acc = ncorrect/val_size * 100
    try:
        recall = npredtrue / ntrue
    except:
        recall = np.nan
    try:
        precision = npredtrue / npred
    except:
        precision = np.nan
    try:
        f1 = 2 / (1/recall + 1/precision)
    except:
        f1 = np.nan
    conf = [[val_size-(ntrue+npred-npredtrue), ntrue-npredtrue], [npred-npredtrue,npredtrue]]
    
    epoch_time = utils.format_time(end-start)
    
    print("")
    print("  epoch time:", epoch_time)
    print("  ncorrect:", ncorrect)
    print("  recall:", recall)
    print("  precision:", precision)
    print("  f1:", f1)
    print("  total_loss:", total_loss)

    print("Average validation accuracy: {0:.4f}".format(acc), "%")
    print("Average validation loss: {0:.4f}".format(avg_loss))

    utils.confusion_matrix(conf)
     
    return acc, avg_loss

    
def build():
    epochs = args['epochs']
    train_acc = [None]*epochs
    train_loss = [None]*epochs
    val_acc = [None]*epochs
    val_loss = [None]*epochs
    
    # For each epoch...
    for epoch in range(epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        
        if args['do_train']:
            print('Training...')
            t_acc, avg_t_loss = training()
            train_acc[epoch] = t_acc
            train_loss[epoch] = avg_t_loss
            
        if args['do_eval']:
            print('\nValidation...')
            v_acc, avg_v_loss  = validation()
            val_acc[epoch] = v_acc
            val_loss[epoch] = avg_v_loss
    
    if kaggle:
        model.save_pretrained('/kaggle/working')
    else:
        model.save_pretrained('models')
        
    return train_acc, val_acc, train_loss, val_loss


# %%
random.seed(args['seed_val'])
np.random.seed(args['seed_val'])
torch.manual_seed(args['seed_val'])
torch.cuda.manual_seed_all(args['seed_val'])


train_acc, validation_acc, train_loss, validation_loss = build() 

print("\n-------------------------------")
if args['do_train']:
    print("Final Training Accuracy {0:.2f}".format(train_acc[-1]), "%")
if args['do_eval']:
    print("Final Validation Accuracy {0:.2f}".format(validation_acc[-1]), "%")
print("-------------------------------")

utils.plot_epochs(args['epochs'], train_acc, validation_acc, train_loss, validation_loss)

