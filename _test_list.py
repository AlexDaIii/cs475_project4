import numpy as np
x = [[1,2,3,4,5,6,7],[1,1,1,1]]

y = np.array([[1,2,3,4]])
yy = np.array([[9,8,7,6]])
# print(y.shape)
y = np.concatenate((y, yy), axis=0)
yy = y[0,:].reshape(1,4)
y = np.concatenate((y, y[0,:].reshape(1,4)), axis=0)
# print(y)


#print(len(x))

# xx = [-1, 0, 0, 2]
# x.append(xx)
# print(x)
# print(x[1])

z = 2*np.ones((900,1))
z = np.mean(z)
print(np.mean(z).size)
zz = 6*np.ones((900, 2))
print(np.divide(zz, z))


