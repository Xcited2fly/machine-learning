from sklearn import tree
# [height, hair-length, voice-pitch]
X = [ [180,15,0],
      [167,42,1],
      [136,35,1],
      [174,15,0],
      [141,28,1]]
y = ['man','woman','woman','man','woman']

clf=tree.DecisionTreeClassifier()
clf=clf.fit(X,y)
prediction=clf.predict([[133,37,1]])
print(prediction)
