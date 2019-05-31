import sys

import re
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def remove_special_characters(line):
    # pattern = '[!"#$%&\(\)*+/:：（）？?;<=>@\[\]^`\{\|\}~\t\n]+'
    pattern = '[^a-zA-Z0-9,.\']+'
    new_line = re.sub(pattern, " ", line.lower())
    return new_line

#tokenize and lowercase the characters
def tokenize(line):
    split_by_quote = line.replace('\'', ' ').replace('.', " . ").replace(',', " , ").replace('-', ' - ').split()
    return split_by_quote
    
def remove_stop_words(tokens):
    length = len(tokens)
    i = 0
    while i < length:
        if tokens[i] in stop_words:
            del tokens[i]
            length -= 1
        else:i += 1
    return tokens



# In[227]:


def pipeline(lines):
    tokens_raw = []
    tokens_without_stopwords = []
    for line in lines:
        tokens = tokenize(remove_special_characters(line))
        tokens_raw.append(tokens)
        tokens_without_stopwords.append(remove_stop_words(tokens))
    train_list, test_val_list = train_test_split(tokens_raw, test_size=0.2)
    test_list, val_list = train_test_split(test_val_list, test_size=0.5)
    train_list_no_stopwords, test_val_list_no_stopwords = train_test_split(tokens_without_stopwords, test_size=0.2)
    test_list_no_stopwords, val_list_no_stopwords = train_test_split(test_val_list_no_stopwords, test_size=0.5)
    np.savetxt("train.csv", train_list, delimiter=",", fmt='%s')
    np.savetxt("val.csv", val_list, delimiter=",", fmt='%s')
    np.savetxt("test.csv", test_list, delimiter=",", fmt='%s')

    np.savetxt("train_no_stopword.csv", train_list_no_stopwords,
               delimiter=",", fmt='%s')
    np.savetxt("val_no_stopword.csv", val_list_no_stopwords,
               delimiter=",", fmt='%s')
    np.savetxt("test_no_stopword.csv", test_list_no_stopwords,
               delimiter=",", fmt='%s')

if __name__ == "__main__":
    input_path = sys.argv[1]

    """
    Tokenize the input file here
    Create train, val, and test sets
    """
    # line = "This is #what&Iwant$with you,the loser.you know don't fail me"
    # res = remove_special_characters(line)
    # print("before: ",line)
    # print("after:", res)
    raw_data = open(input_path, "r")
    lines = raw_data.readlines()
    pipeline(lines)
