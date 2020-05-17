from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from numpy import shape
from sklearn.metrics import log_loss
import numpy as np
def read_data(path):
	with open(path) as f :
		lines = f.readlines()
	lines=[eval(line.strip()) for line in lines]
	x,y=zip(*lines)
	x=np.array(x)
	y=np.array(y)
	return x,y
def curve(x_train,w,w0):
	results=x_train.tolist()
	for i in range(0,100):
		x1=1.0*i/10
		x2=-1*(w[0]*x1+w0)/w[1]
		results.append([x1,x2])
	results=["{},{}".format(x1,x2) for [x1,x2] in results]
	return results

x_train,y_train=read_data("train_data")
x_test,y_test=read_data("test_data")


model=LogisticRegression()
model.fit(x_train,y_train)

print(model.coef_)
print(model.intercept_)

y_pred=model.predict(x_test)

print(y_pred)

y_pred_p = model.predict_proba(x_test)

print(y_pred_p)
 
