def DataProcessor(data):
    '''
    Creation of data dataframes.
    -----------------------------
    input = CSV Path
    output = - Questions DataFrame
             - Target DataFrame
    '''
    df = pd.read_csv(data).fillna("")
    questions_ab = df[["question1","question2"]]
    targets = df[["is_duplicate"]]
    
    return questions_ab, targets

def DataTokenizer(data):
    '''
    Tokenization of data using BertTokenizer
    -----------------------------
    input = Questions dataframe
    output = Tokens dataframe
    '''

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens_ab = data
    
    for i in range(len(data)): 
        tokens_ab.loc[i]["question1"] = [tokenizer.tokenize(data.loc[i]["question1"])]
        tokens_ab.loc[i]["question2"] = [tokenizer.tokenize(data.loc[i]["question2"])]
    
    return tokens_ab
