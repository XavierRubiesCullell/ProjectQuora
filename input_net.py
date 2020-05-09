import utils

def create_input(in_file, out_file):
    questions_ab, targets = utils.DataProcessor(file)
    max_seq_length = 70

    INPUT_IDS, SEGMENT_IDS, POSITION_IDS = DataFeatures(questions_ab,max_seq_length)

    i = pd.DataFrame(INPUT_IDS)
    s = pd.DataFrame(SEGMENT_IDS)
    p = pd.DataFrame(POSITION_IDS)
    total = pd.concat([i, s, p], axis=1)

    WriteCSV(total, out_file)

    return total