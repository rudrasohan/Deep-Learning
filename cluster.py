import numpy as np
import matplotlib.pyplot as plt
import tflearn

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
#print(X)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), pow(-1,j)*r*np.cos(t)]
  y[ix] = j
# lets visualize the data:
#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
#plt.show()

a = y.shape[0]
b = np.zeros((a,3))
b[np.arange(a), y] = 1
b.shape
X.shape
data = X
labels = b

net = tflearn.input_data(shape=[None, 2])
net = tflearn.fully_connected(net, 100, activation='relu')
#net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 3, activation='softmax')
net = tflearn.regression(net,loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.fit(data, labels, n_epoch=1000, batch_size=16, show_metric=True)

h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
#Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()