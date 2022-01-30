import pandas as pd
import numpy as np
import exploratory
import preprocessing
import vectorization
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


import models

df = pd.read_csv("messages.csv")

exploratory.information_of_dataset(df)

exploratory.checking_nulls(df)

df= preprocessing.creating_coverting_new_colums(df)

df= preprocessing.removing_stopwords_and_punctuation(df)

exploratory.notspam_length_diagram(df)

exploratory.spam_length_diagram(df)

X_transform=vectorization.cvectorization(df)

X_tfidf= vectorization.tfidvectorization(X_transform)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['label'], test_size=0.25, random_state = 50)

svcmodel=models.svcmodel(df,X_tfidf,X_train,y_train)

naivemodel=models.naive_bayes(df,X_tfidf,X_train,y_train)


naivepredictions = naivemodel.predict(X_test)
print("-------------------------------------")
print("-----------Naive Bayes-----------------")
print("-------------------------------------")
print(classification_report(y_test, naivepredictions))
print(confusion_matrix(y_test,naivepredictions))
print("-------------------------------------")



svcpredictions = svcmodel.predict(X_test)
print("-------------------------------------")
print("-----------------SVC-----------------")
print("-------------------------------------")
print(classification_report(y_test, svcpredictions))
print(confusion_matrix(y_test,svcpredictions))
print("-------------------------------------")
