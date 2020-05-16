import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import matplotlib.pyplot as plt
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix


# Main functions
def ToTensor(data):
    input_ids = np.array(data.iloc[:,:70])
    token_type_ids = np.array(data.iloc[:,70:140])
    attention_masks = np.array(data.iloc[:,140:210])
    targets = np.array(data.iloc[:,-1])
    
    tokens_tensor = torch.tensor(input_ids, dtype=torch.long)
    segments_tensor = torch.tensor(token_type_ids, dtype=torch.long)
    attention_tensor = torch.tensor(attention_masks, dtype=torch.long)
    targets_tensor = torch.tensor(np.array(targets), dtype=torch.long)

    return tokens_tensor, segments_tensor, attention_tensor, targets_tensor


# Input Net functions
def ReadData(data):
    '''
    Creation of data dataframes.
    -----------------------------
    input = CSV Path

    output =    - Questions DataFrame
                - Target DataFrame
    '''
    df = pd.read_csv(data).fillna("")
    questions_ab = df[["question1","question2"]]
    targets = df[["is_duplicate"]]
    
    return questions_ab, targets


def truncate(tokens_a, tokens_b, max_seq_length):
    '''
    Truncates by max_seq_length the concatenation of the two sentences
    '''
    len_a = len(tokens_a);  len_b = len(tokens_b)
    len_t = len_a + len_b
    m = min(1,max_seq_length/len_t)
    if len_a < len_b:
        return tokens_a[:int(np.ceil(len_a*m))], tokens_b[:int(np.floor(len_b*m))]
    else:
        return tokens_a[:int(np.floor(len_a*m))], tokens_b[:int(np.ceil(len_b*m))]


def token_features(tokens_a, tokens_b):
    '''
    Adds the special tokens: [CLS], [SEP], [SEP].
    Creates the segmentation vector of the two sentences.
    '''
    tokens = ["[CLS]"]
    segment_ids = [0]
    
    #The 1st sentence assings a seg. value of 0 and the 2nd one a value of 1
    tokens += tokens_a
    segment_ids += [0]*len(tokens_a)
    tokens.append("[SEP]")
    segment_ids.append(0)
    
    tokens += tokens_b
    segment_ids += [1]*len(tokens_b)
    tokens.append("[SEP]")
    segment_ids.append(1)
    
    return tokens, segment_ids


def DataFeatures(data, max_seq_length=None):
    '''
    Create phrase embeddings for tokens and sentence segmentation.
    ----------------------------------
    input =     - Questions DataFrame
                - Maximum sequence size

    output =    - inputs_ids
                - segment_ids
                - positional_ids
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    INPUT_IDS = []
    SEGMENT_IDS = []
    POSITION_IDS = []
    for i in range(len(data)):
        question_a = data.loc[i]["question1"]
        question_b = data.loc[i]["question2"]
        tokens_a = tokenizer.tokenize(question_a)
        tokens_b = tokenizer.tokenize(question_b)
        
        if (max_seq_length):
            # Account for [CLS], [SEP], [SEP] with "- 3"
            tokens_a, tokens_b = truncate(tokens_a, tokens_b, max_seq_length-3)
            
        # Add special tokens and generate segment_ids
        tokens, segment_ids = token_features(tokens_a, tokens_b)
        
        # Convert word tokens to indexes
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        INPUT_IDS.append(input_ids)
        SEGMENT_IDS.append(segment_ids)
        POSITION_IDS.append([1]*len(input_ids))

    # Zero-pad up to the sequence length.
    INPUT_IDS = pad_sequences(INPUT_IDS, padding='post', truncating="post",maxlen=max_seq_length)
    SEGMENT_IDS = pad_sequences(SEGMENT_IDS, padding='post', truncating="post",maxlen=max_seq_length)
    POSITION_IDS = pad_sequences(POSITION_IDS, padding='post', truncating="post",maxlen=max_seq_length)
    
    print (len(INPUT_IDS), len(SEGMENT_IDS), len(POSITION_IDS))
    #assert len(INPUT_IDS) == max_seq_length
    #assert len(SEGMENT_IDS) == max_seq_length
    #assert len(POSITION_IDS) == max_seq_length

    return INPUT_IDS, SEGMENT_IDS, POSITION_IDS


def metrics ():
    tn, fp, fn, tp  = confusion_matrix(y_target, y_predict).ravel()

    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1 = 2*tp/(2*tp+fp+fn)
    accuracy = (tp+tn)/(tp+fp+fn+tn)
    
    return recall, precision, f1, accuracy


def WriteCSV(data, file_name):
    '''
    Writes a DataFrame to a CSV
    '''
    data.to_csv(file_name, index=False)

