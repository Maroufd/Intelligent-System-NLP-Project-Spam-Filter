import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def information_of_dataset(df):
    print("-------Head------")
    print(df.head())
    print("-----------------")
    print("------Infromation------")
    print(df.info())
    print("--------------------")
    print("----------Shape----------")
    print(df.shape)
    print("--------------------")
    print("----------Total number of mails----------")
    print(df['label'].value_counts())
    print("--------------------")
    print("----------Number of spam and not spam emails----------")
    print(len(df[df['label']==0]),len(df[df['label']==1]))
    print("--------------------")
    return print("Finished")

def checking_nulls(df):
    print("-------Null values------")
    print(df.isnull().values.any())
    print("-----------------")
    print("------Null values by columns------")
    print(df.isnull().sum())
    print("--------------------")
    return print("Finished")

def notspam_length_diagram(df):
    df=df[df['label']==0]
    df['numberofwords'] = df['message'].apply(lambda x: len(x.split(" ")))
    newdf=df.groupby('numberofwords').size().reset_index(name='counts').sort_values(by=['counts'])
    ax=sns.barplot(newdf['numberofwords'], newdf['counts'] ,color = 'cyan'
            ,ci = None
            )
    ax.set(xticks=np.arange(0,len(newdf),100), yticks=np.arange(1,13,1))
    plt.show()
    return print("Finished")
def spam_length_diagram(df):
    df=df[df['label']==1]
    df['numberofwords'] = df['message'].apply(lambda x: len(x.split(" ")))
    newdf=df.groupby('numberofwords').size().reset_index(name='counts').sort_values(by=['counts'])
    ax=sns.barplot(newdf['numberofwords'], newdf['counts'] ,color = 'red'
            ,ci = None
            )
    ax.set(xticks=np.arange(0,len(newdf),100), yticks=np.arange(1,13,1))
    plt.show()
    return print("Finished")
