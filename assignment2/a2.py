#!/usr/bin/env python
# coding: utf-8

# In[123]:

import scipy
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# In[124]:


def read_data(path_dict):
    df_dict = {}
    for name, path in path_dict.items():
        df_dict[name] = pd.read_csv(path, sep = 'delimiter',header = None)
    return df_dict

def concat_to_sentences(word2d_dict):
    sentences_list_dict = {}
    for name, lst in word2d_dict.items():
        stc_list = []
        for line in lst:
            stc_list.append("".join(line))
            #something = " ".join(line)
            #print(line)
        sentences_list_dict[name] = stc_list
    return sentences_list_dict

def lists_to_n_grams(list_dict, N, M):
    ngram_dict = {}
    #parameters to tune: max_df, min_df, max_features
    vec = CountVectorizer(ngram_range = (N,M))
    vec.fit(list_dict['train'])
    #1. I use another vectorizer instance for list without stopwors. Is it necessary?
    #2. Only learn vocabulary from train list, is it enough?
    vec_no_stopword = CountVectorizer(ngram_range = (N,M))
    vec_no_stopword.fit(list_dict['train_no_stopword'])
    for name, lst in list_dict.items():
        if name.endswith("stopword"):
            ngram_dict[name] = vec_no_stopword.transform(lst)
        else:
            ngram_dict[name] = vec.transform(lst)
    return ngram_dict

#use Tf-Idf to better model the input
#parameters to tune: norm, sublinear_tf


def tfidf_ize(bow_dict):
    tfidf = TfidfTransformer()
    tfidf.fit(bow_dict['train'])
    tfidf_no_stopword = TfidfTransformer()
    tfidf_no_stopword.fit(bow_dict['train_no_stopword'])
    tfidf_dict = {}
    for name, bow in bow_dict.items():
        if "stopword" in name:
            tfidf_dict[name] = tfidf_no_stopword.transform(bow)
        else:
            tfidf_dict[name] = tfidf.transform(bow)
    return tfidf_dict

def train_clf(data, df_dict):
    result = {}
    clf = MultinomialNB()
    clf_no_stopword = MultinomialNB()
    clf.fit(data['train'], df_dict['train']['target'])
    predicted = clf.predict(data['test'])
    accuracy = np.mean(predicted == df_dict['test']['target'])
    result['all'] = accuracy
    
    clf_no_stopword.fit(data['train_no_stopword'], df_dict['train_no_stopword']['target'])
    predicted = clf_no_stopword.predict(data['test_no_stopword'])
    accuracy = np.mean(predicted == df_dict['test_no_stopword']['target'])
    result['no_stopword'] = accuracy
    return result

# In[125]:

dp = "../assignment1/"
ddp = dp +"pos_gen/"
ddn = dp +"neg_gen/"

pos_test_path = ddp + "test.csv"
pos_train_path = ddp + "train.csv"
pos_val_path = ddp + "val.csv"
pos_test_no_stopword = ddp + "test_no_stopword.csv"
pos_train_no_stopword = ddp + "train_no_stopword.csv"
pos_val_no_stopword = ddp + "val_no_stopword.csv"

neg_test_path = ddn+ "test.csv"
neg_train_path = ddn + "train.csv"
neg_val_path = ddn + "val.csv"
neg_test_no_stopword = ddn + "test_no_stopword.csv"
neg_train_no_stopword = ddn + "train_no_stopword.csv"
neg_val_no_stopword = ddn + "val_no_stopword.csv"

path_dict = {'pos_test':pos_test_path,
'pos_train':  pos_train_path,
'pos_val': pos_val_path,
'pos_test_no_stopword': pos_test_no_stopword,
'pos_train_no_stopword': pos_train_no_stopword,
'pos_val_no_stopword': pos_val_no_stopword,
'neg_test': neg_test_path,
'neg_train': neg_train_path,
'neg_val': neg_val_path,
'neg_test_no_stopword': neg_test_no_stopword,
'neg_train_no_stopword': neg_train_no_stopword,
'neg_val_no_stopword': neg_val_no_stopword
            }

df_dict = read_data(path_dict)

# assign labels for sentences.
for name, df in df_dict.items():
    if name.startswith("pos"):
        df['target'] = pd.Series(1, index = df.index)
#         print(df.shape)
    else:
        df['target'] = pd.Series(0, index = df.index)
        
# merge neg dataframe and pos dataframe
merged_df_dict = {}
merged_df_dict['train'] = pd.concat([df_dict['neg_train'],df_dict["pos_train"]]).reset_index(drop=True)
merged_df_dict['val'] = pd.concat([df_dict['neg_val'],df_dict["pos_val"]]).reset_index(drop=True)
merged_df_dict['test'] = pd.concat([df_dict['neg_test'],df_dict["pos_test"]]).reset_index(drop=True)
merged_df_dict['train_no_stopword'] = pd.concat([df_dict['neg_train_no_stopword'],df_dict["pos_train_no_stopword"]]).reset_index(drop=True)
merged_df_dict['test_no_stopword'] = pd.concat([df_dict['neg_test_no_stopword'],df_dict["pos_test_no_stopword"]]).reset_index(drop=True)
merged_df_dict['val_no_stopword'] = pd.concat([df_dict['neg_val_no_stopword'],df_dict["pos_val_no_stopword"]]).reset_index(drop=True)


# In[128]:
train_list = merged_df_dict['train'].iloc[:,0].values.tolist()
train_no_stopword_list = merged_df_dict['train_no_stopword'].iloc[:,0].values.tolist()

val_list = merged_df_dict['val'].iloc[:,0].values.tolist()
val_no_stopword_list = merged_df_dict['val_no_stopword'].iloc[:,0].values.tolist()

test_list = merged_df_dict['test'].iloc[:,0].values.tolist()
test_no_stopword_list = merged_df_dict['test_no_stopword'].iloc[:,0].values.tolist()


# In[133]:
word2d_dict = {"train": train_list,
            "test": test_list,
            "val": val_list,
            "train_no_stopword": train_no_stopword_list,
            "test_no_stopword": test_no_stopword_list,
            "val_no_stopword": val_no_stopword_list}

list_dict = concat_to_sentences(word2d_dict)

# In[136]:
bow_unigram_dict = lists_to_n_grams(list_dict, 1, 1)
bow_hybr_gram_dict = lists_to_n_grams(list_dict, 1, 2)
bow_bigram_dict = lists_to_n_grams(list_dict, 2, 2)

# In[140]:
tfidf_unigram_dict = tfidf_ize(bow_unigram_dict)
tfidf_hybr_gram_dict = tfidf_ize(bow_hybr_gram_dict)
tfidf_bigram_dict = tfidf_ize(bow_bigram_dict)
        
# In[148]:
unigram_acc = train_clf(tfidf_unigram_dict, merged_df_dict)
hybrid_gram_acc = train_clf(tfidf_hybr_gram_dict, merged_df_dict)
bigram_acc = train_clf(tfidf_bigram_dict, merged_df_dict)


# In[149]:

# results before tuning
for acc in [unigram_acc, hybrid_gram_acc, bigram_acc]:
    print(acc.items())


# In[150]:

#parameter tuning
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

text_clf = Pipeline([
#     ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf_no_stopword = Pipeline([
#     ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

#text_clf.fit(bow_unigram_dict['train'], merged_df_dict['train']['target'])


# In[ ]:

parameters = {
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-1, 1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)
best_clf = gs_clf.fit(bow_unigram_dict['val_no_stopword'], merged_df_dict['val_no_stopword']['target'])
gs_clf.best_score_


# In[ ]:
best_clf.best_params_
best_clf.fit(bow_unigram_dict['train_no_stopword'], merged_df_dict['train_no_stopword']['target'])
predicted = best_clf.predict(bow_unigram_dict['test_no_stopword'])
accuracy = np.mean(predicted == merged_df_dict['test_no_stopword']['target'])
print(accuracy)

