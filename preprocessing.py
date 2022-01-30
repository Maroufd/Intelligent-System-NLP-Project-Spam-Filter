import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords


def creating_coverting_new_colums(df):
    df['length'] = df.message.str.len()
    df['message'] = df['message'].str.lower()
    print("-----------New dataframe------------")
    print(df.head())
    print("----------------------------")
    return df


def removing_stopwords_and_punctuation(df):
    stop_words = set(stopwords.words('english'))
    punctuation=string.punctuation
    df['message'] = df['message'].apply(lambda x: " ".join(wordsom for wordsom in x.split() if wordsom not in punctuation))
    df['message'] = df['message'].apply(lambda x: " ".join(wordsom for wordsom in x.split() if wordsom not in stop_words))
    df['new_length'] = df.message.str.len()
    print("-----------New dataframe------------")
    print(df.head())
    print("----------------------------")
    return df
