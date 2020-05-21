import utils
import pandas as pd

def create_input(in_file, out_file, tokenizer, max_seq_length):
    questions_ab, targets = utils.ReadData(in_file)
    #max_seq_length = 70

    INPUT_IDS, SEGMENT_IDS, POSITION_IDS = utils.DataFeatures(questions_ab, tokenizer, max_seq_length)
    # per al writecsv segurament és millor pandas que una llista, peo per a la xarxa nose, ho dubto
    i = pd.DataFrame(INPUT_IDS)
    s = pd.DataFrame(SEGMENT_IDS)
    p = pd.DataFrame(POSITION_IDS)
    total = pd.concat([i, s, p, targets], axis=1)
    total.columns = ["I"+str(i) for i in range(70)] + ["S"+str(i) for i in range(70)] + ["P"+str(i) for i in range(70)] + ["is_duplicate"]
    utils.WriteCSV(total, out_file)

    return total