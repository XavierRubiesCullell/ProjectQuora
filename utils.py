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

def DataProcessor(data):
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
    tok_a = tokens_a[:];  tok_b = tokens_b[:]
    len_a = len(tok_a);  len_b = len(tok_b)
    len_t = len_a + len_b
    m = min(1,max_seq_length/len_t)
    if len_a < len_b:
        return tok_a[:int(np.ceil(len_a*m))], tok_b[:int(np.floor(len_b*m))]
    else:
        return tok_a[:int(np.floor(len_a*m))], tok_b[:int(np.ceil(len_b*m))]


def token_features(tokens_kind, tokens, segment_ids, kind):
    '''
    Adds the special tokens: [CLS], [SEP], [SEP].
    Creates the segmentation vector of the two sentences.
    '''
    
    #The first sentence assings a seg. value of 1
    # and the second one a value of 2
    seg = 1 if kind == "a" else 2
    for token in tokens_kind:
        tokens.append(token)
        segment_ids.append(seg)
    tokens.append("[SEP]")
    segment_ids.append(seg)
    
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
    INPUT_IDS = []
    SEGMENT_IDS = []
    for i in range(len(data)):
        question_a = data.loc[i]["question1"]
        question_b = data.loc[i]["question2"]
        tokens_a = tokenizer.tokenize(question_a)
        tokens_b = tokenizer.tokenize(question_b)
        
        if (max_seq_length):
            # Account for [CLS], [SEP], [SEP] with "- 3"
            tokens_a, tokens_b = truncate(tokens_a, tokens_b, max_seq_length-3)
            
        tokens = ["[CLS]"]
        segment_ids = [1]
        
        # Add special tokens and generate segment_ids
        tokens, segment_ids = token_features(tokens_a, tokens, segment_ids, "a")
        tokens, segment_ids = token_features(tokens_b, tokens, segment_ids, "b")
        
        # Convert word tokens to indexes
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        INPUT_IDS.append(input_ids)
        SEGMENT_IDS.append(segment_ids)
        POSITION_IDS.append(list(range(len(input_ids))))

    # Zero-pad up to the sequence length.
    INPUT_IDS = pad_sequences(INPUT_IDS, padding='post', truncating="post",maxlen=max_seq_length)
    SEGMENT_IDS = pad_sequences(SEGMENT_IDS, padding='post', truncating="post",maxlen=max_seq_length)
    POSITION_IDS = pad_sequences(POSITION_IDS, padding='post', truncating="post",maxlen=max_seq_length)

    assert len(INPUT_IDS) == max_seq_length
    assert len(SEGMENT_IDS) == max_seq_length
    assert len(POSITION_IDS) == max_seq_length

    return INPUT_IDS, SEGMENT_IDS, POSITION_IDS

def WriteCSV(data, file_name):
    '''
    Writes a DataFrame to a CSV
    '''
    data.to_csv(file_name, index=False)
