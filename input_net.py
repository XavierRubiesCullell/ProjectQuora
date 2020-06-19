import utils
import pandas as pd

def create_input(data, tokenizer, pretrained_model, max_len, out_file = None):
    '''
    Create the input model data 
    - input:    data: raw data
                tokenizer: model tokenizer
                pretrained_model: pretrained configuration
                max_len: max question length 
                output_file: CSV filename
    - output:   total: dataframe
    '''

    questions_ab = data[["question1","question2"]]
    targets = data[["is_duplicate"]]

    print("Dataset length:", len(questions_ab))
    tokenizer = tokenizer.from_pretrained(pretrained_model, do_lower_case=True)
    print("Pretrained model used:", pretrained_model)
    print("Maximum sequence length:", max_len)

    INPUT_IDS, SEGMENT_IDS, POSITION_IDS = utils.DataFeatures(questions_ab, tokenizer, max_len)

    i = pd.DataFrame(INPUT_IDS)
    s = pd.DataFrame(SEGMENT_IDS)
    p = pd.DataFrame(POSITION_IDS)

    total = pd.concat([i, s, p, targets], axis=1)
    total.columns = ["I"+str(i) for i in range(max_len)] + ["S"+str(i) for i in range(max_len)] + ["P"+str(i) for i in range(max_len)] + ["is_duplicate"]
    
    if out_file:
        utils.WriteCSV(total, out_file)
        print("Input saved:", out_file)

    return total