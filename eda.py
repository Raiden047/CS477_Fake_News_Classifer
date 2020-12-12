import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dataScrub as ds
import re

from sklearn.model_selection import train_test_split
import sweetviz as sv


data_set = pd.read_csv('data_clean.csv')

def remove(text):
    text = text.replace('trump', '')
    #text = re.sub('trump', '', text) 
    return text

data_set['text'] = data_set['text'].astype(str).apply(remove)

data_text = data_set[data_set.isReal == 0]['text'].astype(str)

plt.figure(figsize = (10, 6))
most_common_uni = ds.get_top_text_ngrams(data_text, 10, 1)
most_common_uni = dict(most_common_uni)
sns.barplot(x=list(most_common_uni.values()),y=list(most_common_uni.keys()))

plt.title("Fake News: Word Frequency")
plt.xlabel("Count")
plt.ylabel("Word")
plt.show()



