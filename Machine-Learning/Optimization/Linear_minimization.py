#%%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from numpy import linalg as LA
from matplotlib import animation


def f(x, y):
    return x**2 + x*y + y**2

x = np.linspace(-2,2,20)
y = np.linspace(-2,2,20)
X, Y = np.meshgrid(x, y)
Z = f(X,Y)

plt.contour(X,Y,Z)

# X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()

fig = plt.figure()

ax = plt.axes(projection ='3d')

ax.plot_surface(X, Y, Z, cmap = plt.cm.CMRmap, edgecolor ='green')

ax.view_init(40, 45)
plt.show()
#%%


def linear_minimization(function, initial_condition, n_iter):
    
    hessian_sym = sp.hessian(function, [x,y])
    hessian  = np.array(hessian_sym, dtype= int)
    
    gradient_ = np.array(([sp.diff(function, x).replace(x, 0).replace(y, 0)], [sp.diff(function, y).replace(x, 0).replace(y, 0)]), dtype=float).reshape(2,1)
    
    stationary_point = -1 * (LA.inv(hessian) @ gradient_)
    
    x_k_history = []
    
    for i in range(n_iter):
    
       gradient = np.array(([sp.diff(function, x).replace(x, initial_condition[0][0]).replace(y, initial_condition[1][0])], [sp.diff(function, y).replace(x, initial_condition[0][0]).replace(y, initial_condition[1][0])]), dtype=float).reshape(2,1)
       
       p = -1 * gradient
       
       alpha = -1 * (gradient.T @ p)/(p.T @ (hessian @ p))
       
       x_k = initial_condition - (alpha * gradient)
       
       initial_condition = x_k 
       
       x_k_history.append(x_k)
       
       if np.abs(np.linalg.norm(gradient)) < 1e-6:
        break
       
    return np.array(x_k_history), stationary_point

x, y = sp.symbols('x y', real=True)

function = x**2 + x*y + y**2

init = np.array([[1], 
                 [-2.5]], 
                dtype='float64')

x_k, stationary = linear_minimization(function, 
                                      init, 
                                      200)
# %%


fig1, ax1 = plt.subplots(figsize = (7,7))
ax1.contour(X, Y, Z, 100, cmap = 'jet')
ax1.set_title(f"Steepest descent of function: {function}")

line, = ax1.plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
point, = ax1.plot([], [], '*', color = 'red', markersize = 4)
value_display = ax1.text(0.02, 0.02, '', transform=ax1.transAxes)

def init_1():
    line.set_data([], [])
    point.set_data([], [])
    value_display.set_text('')

    return line, point, value_display

def animate_1(i):
    # Animate line
    line.set_data(x_k[:i,0], x_k[:i, 1])
    
    # Animate points
    point.set_data(x_k[i, 0], x_k[i, 1])

    return line, point, value_display

ax1.legend(loc = 1)

anim1 = animation.FuncAnimation(fig1, animate_1, init_func=init_1, frames=len(x_k), interval=20, blit=True)


anim1.save('Linear_minimization_contour.gif', writer='ffmpeg', fps =1)
# %%

z_history = np.zeros(len(x_k))

for i in range(len(x_k)):
    z_history[i] = x_k[i][0]**2 + x_k[i][1]**2

z_history = z_history.reshape((len(x_k)))

new_x_k = x_k.reshape((len(x_k) * 2))

x_history = new_x_k[::2]
y_history = new_x_k[1::2]


#%%
fig2 = plt.figure(figsize = (7,7))
ax2 = plt.axes(projection ='3d')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

ax2.plot_surface(X, Y, Z, cmap = plt.cm.CMRmap, edgecolor ='green', alpha = 0.5)

ax2.view_init(40, 165)

line, = ax2.plot([], [], [], 'r', label = 'Gradient descent', lw = 1.5)
point, = ax2.plot([], [], [], '*', color = 'red', markersize = 4)


def init_2():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])

    return line, point, 

def animate_2(i):
    line.set_data(x_history[:i], y_history[:i])
    line.set_3d_properties(z_history[:i])
    # Animate points
    point.set_data(x_history[i], y_history[i])
    point.set_3d_properties(z_history[i])
    return line, point,

ax2.legend(loc = 1)

anim1 = animation.FuncAnimation(fig2, animate_2, init_func=init_2, frames=len(x_k), interval=20,repeat_delay=60, blit=True)
plt.show()

anim1.save('Linear_minimization_surface.gif', writer='ffmpeg', fps =1)


# %%
