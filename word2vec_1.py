
### Imports
import pandas as pd
import numpy as np
import os


train = pd.read_csv('../dataset/L.POINT_train.csv', encoding='UTF-8')
test = pd.read_csv('../dataset/L.POINT_test.csv', encoding='UTF-8')

p_level = 'CLAC3_NM'

def oversample(x, n, seed=0):
    if n == 0:
        return list(x)
    uw = np.unique(x)
    bs = np.array([])
    np.random.seed(seed)
    for j in range(n):
        bs = np.append(bs, np.random.choice(uw, len(uw), replace=False))
    return list(bs)

train_corpus = list(train.groupby('CLNT_ID')[p_level].agg(oversample, 30))
test_corpus = list(test.groupby('CLNT_ID')[p_level].agg(oversample, 30))


num_features = 3
min_word_count = 1 
context = 3 


from gensim.models import word2vec

w2v = word2vec.Word2Vec(train_corpus, 
                        size=num_features, 
                        min_count=min_word_count,
                        window=context,
                        seed=0, workers=1)

w2v.init_sims(replace=True)


class EmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = num_features
    def fit(self, X):
        return self
    def transform(self, X):
        return np.array([
            np.hstack([
                np.max([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),
                np.min([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),
                np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),                
                np.std([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0)                
            ]) 
            for words in X
        ]) 

train_features = pd.DataFrame(EmbeddingVectorizer(w2v.wv).fit(train_corpus).transform(train_corpus))
test_features = pd.DataFrame(EmbeddingVectorizer(w2v.wv).transform(test_corpus))

train_features.columns = ['v'+f'{c+1:03d}' for c in train_features.columns]
test_features.columns = ['v'+f'{c+1:03d}' for c in test_features.columns]

pd.concat([pd.DataFrame({'CLNT_ID': np.sort(train['CLNT_ID'].unique())}), train_features], axis=1).to_csv('X_train_w2v_CLAC3_NM.csv', index=False)
pd.concat([pd.DataFrame({'CLNT_ID': np.sort(test['CLNT_ID'].unique())}), test_features], axis=1).to_csv('X_test_w2v_CLAC3_NM.csv', index=False)
