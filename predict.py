
import pandas as pd
from cleaner import *
import pickle
import warnings
warnings.filterwarnings("ignore")


new_data = pd.DataFrame()



body = input("Enter the body of article: \n")

while len(body) < 100:
    print("Article is not big enough to predict")
    body = input("Enter the article (greater than 100 letters): \n")




new_data['Body'] = [body]

# combine headline and body into text
new_data['Text'] =  new_data['Body']


new_data['semantics'] = semantics(new_data)

model = pickle.load(open('svm_model.pickle','rb'))

prediction = model.predict(new_data['semantics'])
prob1 = model.predict_proba(new_data['semantics'])
print("")
print("prediction from SVM Model:")
print("the news is : "+prediction[0])
print("the news has probablity of:",prob1,"[FALSE : TRUE]")
print("")
model = pickle.load(open('nb_model.pickle','rb'))

prediction1 = model.predict(new_data['semantics'])
prob2 = model.predict_proba(new_data['semantics'])
print("")
print("prediction from Naive Bayes Model:")
print("the news is : "+prediction1[0])
print("the news has probablity of:",prob2,"[FALSE : TRUE]")
print("")




print("Final prediction (SVM + Naive Bayes): ")

fin=prob1[0][1]+prob2[0][1]

if (fin > 1):
    print("the news is : REAL")
else :
    print("the news is : FAKE")



