#Importing Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

#Importing Decision Tree

from sklearn.tree import DecisionTreeClassifier


data=pd.read_csv("/content/Iris (1).csv")
data.head()

dummies=pd.get_dummies(data.iloc[:,-1])
dummies

merge=pd.concat([data,dummies],axis='columns')
merge

final=merge.drop(['Species'],axis='columns')
final.head()

final=final.drop(['Iris-setosa'],axis='columns')
final.head()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
newdata=data
newdata.state=le.fit_transform(newdata.Species)
newdata

data.info()

data.shape

data.columns

x=data.iloc[:,:4]
y=data.iloc[:,-1]
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

model=DecisionTreeClassifier(random_state=2)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)#y_test
print(y_pred)

classifier= DecisionTreeClassifier(random_state=2)
classifier.fit(x_train,y_train)


#Decision Tree

from sklearn import tree
tree.plot_tree(classifier)

#Text Representation

text_represent=tree.export_text(classifier)
print(text_represent)

#Confusion Matrix


from sklearn.metrics import confusion_matrix
conmat=confusion_matrix(y_test,y_pred)
conmat

#Accuracy of the decision Tree


from sklearn.metrics import accuracy_score
accurate=accuracy_score(y_test,y_pred)
accurate

#The accuracy of this classifier is 1.0%


