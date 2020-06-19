import numpy as np 
import pandas as pd 
import os
import torch
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sn
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from matplotlib.collections import QuadMesh
import seaborn as sn



#######################################################################################
# Input creation
#######################################################################################

def ToTensor(X, y):
    n = int(int(X.shape[1])/3)
    input_ids = X[:,:n]
    token_type_ids = X[:,n:-n]
    attention_masks = X[:,-n:]

    tokens_tensor = torch.tensor(input_ids, dtype=torch.long)
    segments_tensor = torch.tensor(token_type_ids, dtype=torch.long)
    attention_tensor = torch.tensor(attention_masks, dtype=torch.long)

    if y is None:
        return tokens_tensor, segments_tensor, attention_tensor
    
    targets_tensor = torch.tensor(y, dtype=torch.long)

    return tokens_tensor, segments_tensor, attention_tensor, targets_tensor


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


def DataFeatures(data, tokenizer, max_seq_length=None):
    '''
    Create phrase embeddings for tokens and sentence segmentation.
    ----------------------------------
    input =     - Questions DataFrame
                - Maximum sequence size

    output =    - inputs_ids
                - segment_ids
                - positional_ids
    '''
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
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

    return INPUT_IDS, SEGMENT_IDS, POSITION_IDS


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def WriteCSV(data, file_name):
    '''
    Writes a DataFrame to a CSV
    '''
    data.to_csv(file_name, index=False)

#######################################################################################
# Class classification
#######################################################################################
def list_from_string(emb):
    emb = emb[1:-1]
    emb = emb.split(', ')
    emb = list(map(lambda x: int(x), emb))
    return emb


def new_nodes(f,s, nodes, classes):
    if f not in nodes:
        nodes[f] = f
        classes[f] = [f]

    if s not in nodes:
        nodes[s] = s
        classes[s] = [s]


def class_join(f,s, nodes, classes):
    class_m = min(nodes[f], nodes[s])
    class_M = max(nodes[f], nodes[s])
    
    classes[class_m] += classes[class_M]
    for node in classes[class_M]:
        nodes[node] = class_m
    del classes[class_M]


def clusters(list_):
    nodes = {}
    classes = {}
    '''
    input: id1, id2, t (target)
    output: it builds classes (dictionary class:[nodes]) and nodes_to_class (dictionary node:class)
    '''
    for f,s,t in list_:
        new_nodes(f,s,nodes,classes)
        if t and nodes[f] != nodes[s]:
            class_join(f,s,nodes,classes)
    return (classes, nodes)

#######################################################################################
# PLOTS
#######################################################################################

# Learning evolution plot
# ················································

def plot_epochs(epochs, train_acc, validation_acc, train_loss, validation_loss):
    x = [i for i in range(1,epochs+1)]

    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 4))
    fig.suptitle('Evolution on epoch index')

    ax1.scatter(x,train_acc)
    ax1.plot(x,train_acc, label="train")
    ax1.scatter(x,validation_acc, color = 'g')
    ax1.plot(x,validation_acc, label="val", color = 'g')
    ax1.set(xlabel='Epochs', ylabel='% of accuracy')
    ax1.set_xticks(x)
    ax1.legend(loc='upper left', shadow=True)

    ax2.scatter(x,train_loss)
    ax2.plot(x,train_loss, label="train")
    ax2.scatter(x,validation_loss, color = 'g')
    ax2.plot(x,validation_loss, label="val", color = 'g')
    ax2.set(xlabel='Epochs', ylabel='Loss')
    ax2.set_xticks(x)
    ax2.legend(loc='lower left', shadow=True)
    plt.show()

# Confusion matrix plot
# ················································

def get_new_fig(fn, figsize=[9,9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()   #Get Current Axis
    ax1.cla() # clear existing plot
    return fig1, ax1


def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = []; text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:,col]
    ccl = len(curr_column)

    #last line  and/or last column
    if(col == (ccl - 1)) or (lin == (ccl - 1)):
        #tots and percents
        if(cell_val != 0):
            if(col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%'%(per_ok), '100%'] [per_ok == 100]

        #text to DEL
        text_del.append(oText)

        #text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d'%(cell_val), per_ok_s, '%.2f%%'%(per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy(); dic['color'] = 'g'; lis_kwa.append(dic);
        dic = text_kwargs.copy(); dic['color'] = 'tab:red'; lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y-0.3), (oText._x, oText._y), (oText._x, oText._y+0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            text_add.append(newText)

        #set background color for sum cells (last line and last column)
        carr = [58/255, 58/255, 58/255, 1.0]
        if(col == ccl - 1) and (lin == ccl - 1):
            carr = [28/255, 28/255, 28/255, 1.0]
        facecolors[posi] = carr

    else:
        if(per > 0):
            txt = '%s\n%.2f%%' %(cell_val, per)
        else:
            if(show_null_values == 0):
                txt = ''
            elif(show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        #main diagonal
        if(col == lin):
            #set color of the textin the diagonal to white
            oText.set_color('w')
            # set background color in the diagonal to blue
            facecolors[posi] = [203/255, 60/255, 60/255, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del


def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append( df_cm[c].sum() )
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append( item_line[1].sum() )
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col

    df_cm.index


def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=14,
      lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='y'):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if(pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    #this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    #thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    #set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    #face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    #iter in text elements
    array_df = np.array( df_cm.to_records(index=False).tolist() )
    text_add = []; text_del = [];
    posi = -1 #from left to right, bottom to top.
    for t in ax.collections[0].axes.texts: #ax.texts:
        pos = np.array( t.get_position()) - [0.5,0.5]
        lin = int(pos[1]); col = int(pos[0]);
        posi += 1

        #set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    #remove the old ones
    for item in text_del:
        item.remove()
    #append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    #titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  #set layout slim
    plt.show()
#

def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="Oranges",
      fmt='.2f', fz=14, lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
    """

    if(not columns):
        from string import ascii_uppercase
        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]

    confm = confusion_matrix(y_test, predictions)
    cmap = 'Oranges';
    fz = 14;
    figsize=[9,9];
    show_null_values = 2
    df_cm = pd.DataFrame(confm, index=columns, columns=columns)
    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values, pred_val_axis=pred_val_axis)


def confusion_matrix(cf):
    array = np.array(cf)
 
    df_cm = pd.DataFrame(array, index=range(1,3), columns=range(1,3))
    cmap = 'PuRd'
    pretty_plot_confusion_matrix(df_cm, cmap=cmap)
