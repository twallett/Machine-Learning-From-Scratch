#%%
import numpy as np 
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return np.exp(2 * (x ** 2)) + (2 * (y ** 2)) + (x) + (-5*y) + 10

x = np.linspace(-1.5,1,20)
y = np.linspace(-1,1,20)
X, Y = np.meshgrid(x, y)
Z = f(X,Y)

plt.contour(X,Y,Z)

#%%
# X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()

fig = plt.figure()

ax = plt.axes(projection ='3d')

ax.plot_surface(X, Y, Z, cmap = plt.cm.CMRmap, edgecolor ='green')

ax.view_init(20, 45)
plt.show()

# %%

def f(x, y):
    return (4*X +1) * np.exp((2 * (X ** 2)) + (2 * (Y ** 2)) + (X) + (-5*Y) + 10)

x = np.linspace(-1.5,1,20)
y = np.linspace(-1,1,20)
X, Y = np.meshgrid(x, y)
Z = f(X,Y)

plt.contour(X,Y,Z)

# %%
fig = plt.figure()

ax = plt.axes(projection ='3d')

ax.plot_surface(X, Y, Z, cmap = plt.cm.CMRmap, edgecolor ='green')

ax.view_init(20, 45)
plt.show()

# %%

def f(x, y):
    return (4*X - 5) * np.exp((2 * (X ** 2)) + (2 * (Y ** 2)) + (X) + (-5*Y) + 10)

x = np.linspace(-1,1,20)
y = np.linspace(-1,1,20)
X, Y = np.meshgrid(x, y)
Z = f(X,Y)

plt.contour(X,Y,Z)

# %%
fig = plt.figure()

ax = plt.axes(projection ='3d')

ax.plot_surface(X, Y, Z, cmap = plt.cm.CMRmap, edgecolor ='green')

ax.view_init(10, 45)
plt.show()

# %%

x = np.linspace(0,0.5,20)
y = x**4 -1/2*x**3 + 1

plt.plot(x,y)

# %%

def f(x, y):
    return (y-x)**4 + (8*x*y) - x + y + 3

x = np.linspace(-1,1,20)
y = np.linspace(-1,1,20)
X, Y = np.meshgrid(x, y)
Z = f(X,Y)

plt.contour(X,Y,Z)

#%%
# X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()

fig = plt.figure()

ax = plt.axes(projection ='3d')

ax.plot_surface(X, Y, Z, cmap = plt.cm.CMRmap, edgecolor ='green')

ax.view_init(40, 45)
plt.show()
# %%


def f(x, y):
    return 5*x**2 + 2*x*y - 2*x - y + y**2

x = np.linspace(-2,2,20)
y = np.linspace(-2,2,20)
X, Y = np.meshgrid(x, y)
Z = f(X,Y)

plt.contour(X,Y,Z)

#%%
# X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()

fig = plt.figure()

ax = plt.axes(projection ='3d')

ax.plot_surface(X, Y, Z, cmap = plt.cm.CMRmap, edgecolor ='green')

ax.view_init(40, 45)
plt.show()

# %%

def f(x, y):
    return x**2+2*y**2

x = np.linspace(-2,2,200)
y = np.linspace(-2,2,200)
X, Y = np.meshgrid(x, y)
Z = f(X,Y)

plt.contour(X,Y,Z)


#%%
# X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()

fig = plt.figure()

ax = plt.axes(projection ='3d')

ax.plot_surface(X, Y, Z, cmap = plt.cm.CMRmap, edgecolor ='green')

ax.view_init(40, 45)
plt.show()

from nndesigndemos import nndtoc

nndtoc()
# %%
