#%%
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff
from numpy import linalg as LA



def f(x, y):
    return 5*x**2 - 6*x*y + 5*y**2 + 4*x + 4*y

x = np.linspace(-2,2,20)
y = np.linspace(-2,2,20)
X, Y = np.meshgrid(x, y)
Z = f(X,Y)

plt.contour(X,Y,Z)

# X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()

fig = plt.figure()

ax = plt.axes(projection ='3d')

ax.plot_surface(X, Y, Z, cmap = plt.cm.CMRmap, edgecolor ='green')

ax.view_init(20, 45)
plt.show()

# %%

f = 5*x**2 - 6*x*y + 5*y**2 + 4*x + 4*y

x, y = symbols('x y', real = True)

hessian  = np.matrix(((diff(f,x,x),diff(f,x,y)), (diff(f,y,x),diff(f,y,y)))).astype(int)

values, vectors = LA.eig(hessian)
print(values, vectors[0], vectors[1])

gradient = np.matrix([[diff(f,x).replace(x,0).replace(y,0)], 
                      [diff(f,y).replace(x,0).replace(y,0)]])

stationary_point = -1 * np.dot((LA.inv(hessian)), gradient)
print(stationary_point)

# %%
# xk+1 = xk - alpha * gradient

# def gradient_descent(xk, alpha):

alpha = 0.01 

x_k = 