import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
def curce_data(x,y,y_pred):
	x = x.tolist()
	y = y.tolist()
	y_pred = y_pred.tolist()
	results=zip(x,y,y_pred)
	results=["{},{},{}".format(s[0][0],s[1][0],s[2][0]) for s in results]
	return results

def read_data(path):
	with open(path) as f :
		lines=f.readlines()
	lines=[eval(line.strip()) for line in lines]
	x,y=zip(*lines)
	x=np.array(x)
	y=np.array(y)
	return x,y
x_train,y_train=read_data("train_data")
x_test,y_test=read_data("test_data")
model = LinearRegression()
model.fit(x_train,y_train)

print(model.coef_)
print(model.intercept_)
y_pred_train = model.predict(x_train)
train_mse = metrics.mean_squared_error(y_train,y_pred_train)
print("训练集MSE",train_mse)

y_pred_test = model.predict(x_test)
test_mse = metrics.mean_squared_error(y_test,y_pred_test)
print("测试集MSE",test_mse)
train_curve=curce_data(x_train,y_train,y_pred_train)
test_curve=curce_data(x_test,y_test,y_pred_test)

with open("train_curve.cvs","w") as f :
	f.writelines("\n".join(train_curve))
with open("test_curve.cvs","w") as f :
	f.writelines("\n".join(test_curve))
