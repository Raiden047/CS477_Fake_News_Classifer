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
def getData(data_file):
    data_set = pd.read_csv(data_file)

    print('Data Set Shape: ', data_set.shape)

    return data_set

import string
# preping text cleaning
toktok = ToktokTokenizer()
stop_Words = set(stopwords.words("english"))
porter = PorterStemmer()
#sno = SnowballStemmer('english')
lem = WordNetLemmatizer()
def scrubData(text):
    # remove numbers
    text = re.sub(r'\d+', '', text) 

    # remove html
    text = BeautifulSoup(text, "html.parser").get_text()

    #Removing the square brackets
    text = re.sub('\[[^]]*\]', '', text)

    # Removing URL's
    text = re.sub(r'http\S+', '', text)

    # remove punctuation 
    translator = str.maketrans('', '', string.punctuation) 
    text.translate(translator) 

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
    words = [lem.lemmatize(word) for word in words]

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

def plotWordCloud(data_set):
    from wordcloud import WordCloud, STOPWORDS
    plt.figure(figsize = (20,20)) # Text that is not Fake
    wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(data_set[data_set.isReal == 0].text))
    plt.imshow(wc , interpolation = 'bilinear')
    plt.show()

# Function to create weight matrix from word2vec gensim model
def get_weight_matrix(model, vocab, EMBEDDING_DIM):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = model[word]
    return weight_matrix


from sklearn.feature_extraction.text import CountVectorizer
def get_top_text_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def removeTrump(text):
    print(text)
    text = text.replace('trump', '')
    text = text.replace('said', '')
    text = text.replace('president', '')
    text = text.replace('people', '')
    text = text.replace('one', '')
    #text = re.sub('trump', '', text) 
    return text

def removeSaid(text):
    text = text.replace('said', '')
    #text = re.sub('trump', '', text) 
    return text

