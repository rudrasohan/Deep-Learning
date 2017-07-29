"""Softmax"""

scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(X):
	a = np.exp(X)
	c = a.sum(axis=0)
	d = a/c
	return d
	pass

print (softmax(scores))

import matplotlib.pyplot as plt
x = np.arange(-2.0,6.0,0.1)
scores = np.vstack([x, np.ones_like(x), 0.2*np.ones_like(x)])
print scores

plt.plot(x, softmax(scores).T , linewidth = 2)
plt.show()	