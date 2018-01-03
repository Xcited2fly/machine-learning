from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.externals import joblib

iris=load_iris()

#Store the feature matrix (X) and response vector (y)
X=iris.data
y=iris.target

#Splitting the feature matrix (X) and response vector (y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=1)

#training model on training data
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

#making predictions on test data
predictions=knn.predict(X_test)

#comparing values
print('kNN model accuracy:',metrics.accuracy_score(y_test,predictions))

#creating out of samples data for predictions
sample=[[2,4,2,1],[2,5,4,3]]

#making predictions
preds=knn.predict(sample)
pred_species=[iris.target_names[p] for p in preds]
print('Predictions:',pred_species)

#Saving the model for future predictions
joblib.dump(knn,'Iris_knn.pkl')
