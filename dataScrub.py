import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import re
import time

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import ToktokTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# extarcts the dataset from the files
def getData(fake_file, true_file):
    fake_set = pd.read_csv(fake_file)
    true_set = pd.read_csv(true_file)

    print('Fake data Set Shape: ',fake_set.shape)
    print('True Set Shape: ',true_set.shape)

    return fake_set, true_set

# preping text cleaning
toktok = ToktokTokenizer()
stop_Words = set(stopwords.words("english"))
porter = PorterStemmer()
#sno = SnowballStemmer('english')
lem = WordNetLemmatizer()

def scrubData(text):
    # remove html
    text = BeautifulSoup(text, "html.parser").get_text()

    #Removing the square brackets
    text = re.sub('\[[^]]*\]', '', text)

    # Removing URL's
    text = re.sub(r'http\S+', '', text)

    # split string into word tokens and lowercase
    word_tokens = toktok.tokenize(text)
    words = [word for word in word_tokens if word.isalpha()]
    words = [token.lower() for token in words]
    
    # remove stopwords
    words = [w for w in words if not w in stop_Words]

    # stemming each token word
    #words = [porter.stem(word) for word in words]
    #words = [sno.stem(word) for word in words]
    
    # lemmatizing each token word
    #words = [lem.lemmatize(word) for word in words]

    # Join back all the word tokens into one string 
    text_review = ''
    for word in words:
        text_review += str(word) + ' '

    text = text_review.lower()
    return text


def TFIDF_vectorize(training_set, fNum):
    start = time.time()

    tfidf = TfidfVectorizer(norm = 'l2', max_features=fNum)
    
    training_set = tfidf.fit_transform(training_set)
    training_set = training_set.toarray()
    
    print('(TFIDF) # of features: ', len(tfidf.get_feature_names()),
     '\t| data new shape: ', training_set.shape,
     '\t| time: ', round(time.time() - start, 2), 's')

    return training_set
    '''
    word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
    word2tfidf = {k: v for k, v in sorted(word2tfidf.items(), key=lambda item: item[1])}
    for word, score in word2tfidf.items():
        print(word, score)
    exit()
    '''

#prints various infomration on the dataset
def data_info(training_set):
    print(training_set.info())
    #print(training_set.describe())
    print(training_set['credit'].value_counts())
    #sns.countplot(training_set['credit'])
    plt.show() # clear imbalance: good creit examples = 0.2408 (7841/32561)

    # Number of unique classes in each object column
    print(training_set.select_dtypes('object').apply(pd.Series.nunique))



