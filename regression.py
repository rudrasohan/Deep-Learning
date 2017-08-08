import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

#read data
dataframe = pd.read_csv('challenge_dataset.txt',sep = ',',header = None)
#print dataframe

x_values = dataframe[[0]].values #converted into an ndarray
y_values = dataframe[[1]].values
#print x_values
#print y_values

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
c = abs(y_values - body_reg.predict(x_values))
#print c
J = np.sum(c)
print 'ErrorCost = %s' %J
plt.show()
