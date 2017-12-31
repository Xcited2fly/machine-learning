from sklearn.linear_model import LinearRegression
X=[[0.17],[0.19],[0.235],[0.3],[0.42]]
y=[[0.63],[0.7],[0.88],[1.5],[7.3]]
model=LinearRegression()
model.fit(X,y)
X_test=[[0.22],[0.35],[0.85]]
y_test=[[0.82],[4.4],[11.3]]
predictions=model.predict(X_test)
for i,p in enumerate(predictions):
    print('Predicted :%s, target :%s'%(p,y_test[i]))
print('R-squared: %.3f'%model.score(X_test,y_test)
