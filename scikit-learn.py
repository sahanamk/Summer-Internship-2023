import sklearn
import pandas as pd
import numpy as np

'''KNN'''

from sklearn.datasets import load_iris
iris=load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)

df['Target']=iris.target
#0 is Iris Setosa
#1 is Iris Versicolour
#2 is Iris Virginica

# Separating the dependent and independent features  
X=df.iloc[:,:-1].values#inputs
y=df.iloc[:,-1].values#target

#splitting into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

print(X_train.shape)
print(y_train.shape)

from sklearn.neighbors import KNeighborsClassifier

# Create KNN classifier
classifier_knn=KNeighborsClassifier(n_neighbors=4)
# Fit the classifier to the data
classifier_knn.fit(X_train, y_train)

y_pred=classifier_knn.predict(X_test)

from sklearn import metrics
# Finding accuracy by comparing actual response values(y_test) with predicted response value(y_pred)  
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
#check accuracy of our model on the test data
classifier_knn.score(X_test, y_test)

# Providing sample data and the model will make predictions out of that data  
sample = [[5, 5, 3, 2], [1, 10, 3, 5],[3,2,5,6]]  #sample data of three flowers
preds=classifier_knn.predict(sample)
pred_species = [iris.target_names[p] for p in preds] 
print("Predictions:", pred_species)  


'''LINEAR MODELLING - LINEAR REGRESSION'''

'''CLUSTERING METHOD'''

#KMeans
















