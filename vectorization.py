from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def cvectorization(df):
    vectorization = CountVectorizer(analyzer = 'word')
    X = vectorization.fit(df['message'])
    X_transform = X.transform(df['message'])
    return X_transform

def tfidvectorization(transform):
    tfidf_trans = TfidfTransformer().fit(transform)
    X_tfidf = tfidf_trans.transform(transform)
    return X_tfidf
