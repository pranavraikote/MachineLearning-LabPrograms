#Import Modules
import pandas as pd

#Read Dataset and store in DataFrame
col =['Age','Gender','FamilyHist','Diet','LifeStyle','Cholesterol','HeartDisease']
data = pd.read_csv('GNB.csv',names =col )
print(data)

#Label Encoding of Data
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for i in range(len(col)):
    data.iloc[:,i] = encoder.fit_transform(data.iloc[:,i])

#Framing of data into X and Y
X = data.iloc[:,0:6]
y = data.iloc[:,-1]

#Splitting of Data for Training and Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Importing the Gaussian NB Classifier and training
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#Accuracy and Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
print('Accuracy :',accuracy_score(y_test, y_pred))
print('Confusion matrix :')
print(confusion_matrix(y_test, y_pred))
