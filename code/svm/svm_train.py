from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV
def read_data(path):
	with open(path) as f:
		lines=f.readlines()
	lines=[eval(line.strip()) for line in lines]
	x,y=zip(*lines)
	x=np.array(x)
	y=np.array(y)
	return x,y
x_train,y_train=read_data("train_data")
x_test,y_test=read_data("test_data")

model = svm.SVC()
model.fit(x_train,y_train)
print(model.support_vectors_)
print(model.support_)
print(len(model.support_))
score = model.score(x_test,y_test)

print(score)
