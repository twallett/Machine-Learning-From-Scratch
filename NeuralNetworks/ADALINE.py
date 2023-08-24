#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation 

# -------------------------------------------------
p = np.array([[1, 1, 2, 2, -1, -2, -1, -2],
              [1, 2, -1, 0, 2, 1, -1, -2]])

t = np.array([[-1, -1, -1, -1, 1, 1, 1, 1],
              [-1, -1, 1, 1, -1, -1, 1, 1]])
# -------------------------------------------------
# Plot the patterns and the targets

plt.scatter(p[0, :2], p[1, :2], marker='s')
plt.scatter(p[0, 2:4], p[1, 2:4], marker='o')
plt.scatter(p[0, 4:6], p[1, 4:6], marker='x')
plt.scatter(p[0, 6:8], p[1, 6:8], marker='P')
plt.ylabel('X2')
plt.xlabel('X1')
plt.title('X2 vs. X1')
plt.show()

#%%

# -------------------------------------------------
# Adaline network and use LMS

def adaline(inputs, labels, alpha, n_iter):
    inputs = np.concatenate([inputs] * n_iter, axis=1)
    labels = np.concatenate([labels] * n_iter, axis=1)
    w_init = np.identity(2)
    b_init = np.array([1, 1]).reshape((2, 1))
    error_l = []
    w_history = []
    b_history = []
    for i in range(1, inputs.shape[1] + 1):
        error = labels[:, i - 1:i] - (w_init @ inputs[:, i - 1:i] + b_init)
        w_init = w_init + 2 * alpha * error @ inputs[:, i - 1:i].T
        b_init = b_init + 2 * alpha * error
        error_l.append(error)
        w_history.append(w_init)
        b_history.append(b_init)
        if error[0] == 0 and error[1] == 0:
            break
        else:
            continue
    return w_history, b_history, w_init, b_init, error_l

w_h, b_h, w, b, e_list = adaline(p, t, 0.04, 50)

#%%

# -------------------------------------------------
# Calculate the sum square error and plot it

e_list_sq = np.array(e_list) ** 2

fig= plt.figure() 

axis = plt.axes(xlim =(0, len(e_list_sq)), 
                ylim =(-2, 4))

line, = axis.plot([], [], lw = 3) 
line2, = axis.plot([], [], lw = 3, label='y2')

def init(): 
    line.set_data([], [])
    line2.set_data([], [])
    return line, line2,

def animate(i):
    
    x = np.linspace(1, i, i)
    
    y = e_list_sq[:i, 0]
    y2 = e_list_sq[:i, 1]
    
    line.set_data(x, y)
    line2.set_data(x, y2)
    
    return line, line2

plt.legend(['X1', 'X2'])
plt.ylabel("Sum of squared errors")
plt.xlabel("Iterations")
plt.title("Sum of squared errors plot")
plt.show()

anim = FuncAnimation(fig, animate, init_func = init, frames=len(e_list_sq))

anim.save('ADALINE_sumofsquarederror.gif', writer='ffmpeg', fps =20)

plt.legend(['X1', 'X2'])
plt.ylabel("Sum of squared errors")
plt.xlabel("Iterations")
plt.title("Sum of squared errors plot")
plt.show()

# %%

# -------------------------------------------------
# Plot the Decision Boundaries and the patterns.

fig, ax = plt.subplots()

plt.scatter(p[0, :2], p[1, :2], marker='s')
plt.scatter(p[0, 2:4], p[1, 2:4], marker='o')
plt.scatter(p[0, 4:6], p[1, 4:6], marker='x')
plt.scatter(p[0, 6:8], p[1, 6:8], marker='P')

ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])

quiver_1 = ax.quiver(0, 0, 0, 0)
quiver_2 = ax.quiver(0, 0, 0, 0)

def animate(i):
    
    ax.clear()
    
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    
    plt.scatter(p[0, :2], p[1, :2], marker='s')
    plt.scatter(p[0, 2:4], p[1, 2:4], marker='o')
    plt.scatter(p[0, 4:6], p[1, 4:6], marker='x')
    plt.scatter(p[0, 6:8], p[1, 6:8], marker='P')
    
    w_h1, w_h2 = w_h[i][0], w_h[i][1]
    b_h1, b_h2 = b_h[i][0], b_h[i][1]
    
    db1_slope = (-w_h1[0]) / (w_h1[1])
    db2_slope = (-w_h2[0]) / (w_h2[1])
    
    db1_int = (-b_h1[0] / w_h1[1])
    db2_int = (-b_h2[0] / w_h2[1])
    
    quiver_1.set_UVC(np.array([w_h1[0]]), np.array([w_h1[1]]))
    ax.add_artist(quiver_1)
    ax.axline((0, db1_int), slope=db1_slope)
    
    quiver_2.set_UVC(np.array([w_h2[0]]), np.array([w_h2[1]]))
    ax.add_artist(quiver_2)
    ax.axline((0, db2_int), slope=db2_slope)
    
    plt.ylabel('X2')
    plt.xlabel('X1')
    plt.title('X2 vs. X1')
    
    return quiver_1, quiver_2,

anim = FuncAnimation(fig, animate, frames=160)

anim.save('ADALINE_classification.gif', writer='ffmpeg', fps =10)

plt.ylabel('X2')
plt.xlabel('X1')
plt.title('X2 vs. X1')

# %%
