#import modulea
import numpy as np

#Import dataset
from sklearn.datasets import load_iris 
iris=load_iris()

#Data and Target Labels
x=iris.data 
y=iris.target

#Splitting of Data for Training and Testing
from sklearn.model_selection import train_test_split 
xtrain,xtest,ytrain,ytest =train_test_split(x,y,test_size=0.4,random_state=1)

#Importing the KNN Classifier and training
from sklearn.neighbors import KNeighborsClassifier 
knn=KNeighborsClassifier(n_neighbors=1) 
knn.fit(xtrain,ytrain)
pred=knn.predict(xtest)

#Accuracy Metrics
from sklearn import metrics 
print("Accuracy",metrics.accuracy_score(ytest,pred))

#Printing all Prediction and Actual Values
ytestn=[iris.target_names[i] for i in ytest]
predn=[iris.target_names[i] for i in pred]

print("  Predicted     Actual")
for i in range(len(pred)):
    print("  ",predn[i],"   ",ytestn[i]) 
