import imp
import numpy as np
from sklearn.neural_network import MLPClassifier

X=np.array([[0,0],[0,1],[1,0],[1,1]])

Y=np.array([1,0,1,0])

#print(X,Y)
model = MLPClassifier(hidden_layer_sizes=(50,), activation = 'logistic', solver = 'adam', max_iter=10000, verbose=10)

model.fit(X,Y)
predict=model.predict(X)
print(predict)
print(model.score(X,Y))
