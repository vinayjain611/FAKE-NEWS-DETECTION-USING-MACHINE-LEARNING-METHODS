import pandas as pd
import pickle
from cleaner import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import *

import matplotlib.pyplot as plt

# data import

# dataset 1
df = pd.read_csv("data.csv");

# rename labels from 0 to 'REAL' and 1 to 'FAKE'
df.loc[df['Label']== 1, 'Label'] = 'REAL'
df.loc[df['Label']== 0, 'Label'] = 'FAKE'

# drop the column URLs from table
df.drop(['URLs'], axis = 1, inplace = True)


# dataset 2

df1 = pd.read_csv("fake.csv");

# select only fake news 
df1 = df1.loc[df1['type']=='fake']
df1.loc[df1['type'] == 'fake', 'type'] = 'FAKE'

# selecting only title, text and type and renaming them
df1 = df1[['title', 'text', 'type']]
df1.columns = ['Headline', 'Body', 'Label']


# dataset 3

df2 = pd.read_csv("news.csv")

# select only columns title, text and label
df2 = df2[['title', 'text', 'label']]

# rename the column names
df2.columns = ['Headline', 'Body', 'Label']


# dataset 4

df3 = pd.read_csv("train.csv")


# select only title, text and label columns
df3 = df3[['title', 'text', 'label']]

# rename the columns
df3.columns = ['Headline', 'Body', 'Label']

# rename the labels names from 0 and 1 to REAL and FAKE
df3.loc[df3['Label'] == 0, 'Label'] = 'REAL'
df3.loc[df3['Label'] == 1, 'Label'] = 'FAKE'


# combining all the four datasets into one

df = df.append(df1, ignore_index = True)
df = df.append(df2, ignore_index = True)
df = df.append(df3, ignore_index = True)
print("adding all four data sets")
print(df.shape)


# drop duplicate rows if present

df = df.drop_duplicates()
print("dropping all duplicates ")
print(df.shape)


# drop articles which has length is less than 10 if present

i = 0
ind = []

for article in df['Body']:
    if len(str(article)) < 10:
        ind.append(i)
    i = i + 1
    
df = df.drop(df.index[ind])
print("after dropping articles whose size is less than 10")
print(df.shape)


# drop null values if present

print("after dropping null values")
df.dropna(inplace=True)

print(df.shape)


# combining headline and article

df['Text'] = df['Headline'] + " " + df['Body']

#print("df.text")
#print(df.head())


df['Semantics'] = semantics(df)




y = df['Label'].astype('str')
#df.to_csv('aftersemantics.csv') 
X_train, X_test, y_train, y_test = train_test_split(df['Semantics'], y, test_size = 0.2)





pipeline1 = Pipeline([('tfidf',TfidfVectorizer(stop_words='english')),
                    ('nbmodel',MultinomialNB(alpha=0.1))])


#train
pipeline1.fit(X_train, y_train)


#predict
pred1 = pipeline1.predict(X_test)




# report

accuracy = accuracy_score(y_test, pred1)

print('Naive Bayes with semantic analysis:')
print('Accuracy is: %0.3f' % accuracy)
print('report')
print(classification_report(y_test, pred1))

#confusion matrix
cm = confusion_matrix(y_test,pred1)
plot_confusion_matrix(pipeline1, X_test, y_test)
plt.title('Naive Bayes with semantic analysis\nConfusion Matrix')
plt.show()  


with open('nb_model.pickle','wb') as handle:
    pickle.dump(pipeline1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print("model saved")


pipeline2 = Pipeline([('tfidf',TfidfVectorizer(stop_words='english')),
                    ('svmmodel',SVC(kernel='linear',gamma='auto', probability=True))])


#train
pipeline2.fit(X_train, y_train)


#predict
pred2 = pipeline2.predict(X_test)



                                     



# report

accuracy = accuracy_score(y_test, pred2)
print('SVM with semantic analysis:')
print('Accuracy is: %0.3f' % accuracy)
print('report')
print(classification_report(y_test, pred2))


#confusion matrix
cm = confusion_matrix(y_test,pred2)
plot_confusion_matrix(pipeline2, X_test, y_test)
plt.title('SVM \nConfusion Matrix')
plt.show()  



with open('svm_model.pickle','wb') as handle:
    pickle.dump(pipeline2, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("model saved")



