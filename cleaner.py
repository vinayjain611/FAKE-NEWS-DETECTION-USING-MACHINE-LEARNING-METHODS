import numpy as np
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from empath import Empath

def clean_text(text):
    # remove punctuation
    text = re.sub('['+string.punctuation+']', '', text)
    text = re.sub(r"[-()\"#/@’;:<>{}`+=~|.!?,]", "", text)
    
    # convert words to lower case and split
    text = text.lower().split()
    
    # remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if w not in stops]
    text = " ".join(text)
    
    # remove all non english and numbers etc.
    text = re.sub(r'[^a-zA-Z\s]', u'', text, flags=re.UNICODE)
    
    # lemmatizing
    text = text.split()
    l = WordNetLemmatizer()
    lemmatized_words = [l.lemmatize(word) for word in text]
    text = " ".join(lemmatized_words)
    
    return text

def semantics(df):
    
    df['Clean_Text'] = df['Text'].apply(lambda x: clean_text(x))
    df.dropna(inplace=True)

    lexicon = Empath()
    semantic = []

    # adding it to respective categories

    for article in df['Clean_Text']:
        d = lexicon.analyze(article, normalize=False)
        x = []
        for key, value in d.items():
            x.append(value)
        x = np.asarray(x)
        semantic.append(x)
    df['Semantic'] = semantic

    categories = []
    a = lexicon.analyze("")
    for key, value in a.items():
        categories.append(key)
    categories

    # replacing test with categories
    sem = []
    for i in range(df.shape[0]):
        a = []
        for j in range(len(semantic[0])):
            for k in range(int(semantic[i][j])):
                a.append(categories[j])
        b = " ".join(a)
        sem.append(b)
    df['Semantics'] = sem
    data = df['Semantics']

    return data
