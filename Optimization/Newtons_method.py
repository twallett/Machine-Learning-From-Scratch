#%%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from numpy import linalg as LA
from matplotlib import animation


def f(x, y):
    return x**2 + y**2 

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


def newtons_method(function, initial_condition, n_iter):
    
    hessian_sym = sp.hessian(function, [x,y])
    hessian  = np.array(hessian_sym, dtype= int)
    
    gradient = np.array([sp.diff(function, x).replace(x,0).replace(y,0), sp.diff(function, y).replace(x,0).replace(y,0)], dtype=float).reshape(2,1)
    
    stationary_point = -1 * (LA.inv(hessian) @ gradient)
    
    x_k_history = []
    
    for i in range(n_iter):
        
       x_k = initial_condition - (LA.inv(hessian) @ gradient)
    
       gradient = np.array(([sp.diff(function, x).replace(x, x_k[0][0]).replace(y, x_k[1][0])], [sp.diff(function, y).replace(x, x_k[0][0]).replace(y, x_k[1][0])]), dtype=float).reshape(2,1)
       
       initial_condition = x_k
       
       x_k_history.append(x_k)
       
       if np.abs(np.linalg.norm(gradient)) < 1e-6:
        break
       
    return np.array(x_k_history), stationary_point

x, y = sp.symbols('x y', real=True)

function = x**2 + y**2
init = np.array([[1], 
                 [1.5]], dtype='float64')

x_k, stationary = newtons_method(function, 
                                 init, 
                                 200)

# %%


fig1, ax1 = plt.subplots(figsize = (7,7))
ax1.contour(X, Y, Z, 100, cmap = 'jet')
ax1.set_title(f"Newtons method of function: {function}")

line, = ax1.plot([], [], 'r', label = 'Newtons method', lw = 1.5)
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


anim1.save('Newtons_method_contour.gif', writer='ffmpeg', fps =1)
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
ax2.set_title(f"Newtons method of function: {function}")

ax2.view_init(40, 165)

line, = ax2.plot([], [], [], 'r', label = 'Newtons method', lw = 1.5)
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

anim1.save('Newtons_method_surface.gif', writer='ffmpeg', fps =1)


# %%
