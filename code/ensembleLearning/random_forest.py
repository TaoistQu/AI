from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
#print(wine.target)

x_train,x_test,y_train,y_test = train_test_split(wine.data,wine.target,test_size=0.3)

rfc=RandomForestClassifier(random_state=0,n_estimators=10)
rfc= rfc.fit(x_train,y_train)
print(y_test)
print(rfc.score(x_test,y_test))
print(rfc.predict(x_test))
