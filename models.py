from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def svcmodel(df,X_tfidf,X_train,y_train):
    clf = SVC(kernel='linear').fit(X_train, y_train)
    return clf


def naive_bayes(df,X_tfidf,X_train,y_train):
    clf = MultinomialNB().fit(X_train, y_train)
    return clf
