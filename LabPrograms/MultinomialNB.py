#Import Modules
import pandas as pd

#Read Dataset and store in DataFrame
msg=pd.read_csv('MNB.txt',names=['message','label'])
msg['labelnum']=msg.label.map({'pos':1,'neg':0})

#X and Y Classes
X=msg.message
y=msg.labelnum
#print(X)
#print(y)

#Splitting of Data for Training and Testing
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,y)

#CountVectorizer and Bag of Words Modelling
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm=count_vect.transform(xtest)

#Importing the Multinomial NB Classifier and training
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_dtm,ytrain)
predicted = clf.predict(xtest_dtm)

#Accuracy and Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
print('Accuracy metrics of the Multinomial NB Classifier :')
print('Accuracy ', accuracy_score(ytest,predicted))
print('Confusion matrix :')
print(confusion_matrix(ytest,predicted))
print('Recall : ', recall_score(ytest,predicted)
print('Precision : ', precision_score(ytest,predicted)
