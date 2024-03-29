#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

def transfer_function(n):
    if n < 0:
        return 0
    else:
        return 1
    
def perceptron(inputs, labels, n_iter):
    inputs = np.array(inputs * n_iter)
    labels = labels * n_iter
    w_init = np.random.rand(1,2).round(1)
    b_init = 0
    error_list = []
    weight_list = []
    bias_list = []
    for i in range(len(inputs)):
        n = np.dot(w_init, inputs[i]) + b_init #n = W*p + b
        a = transfer_function(n) #a = F(n)
        error = labels[i] - a
        error_list.append(error)
        weight_list.append(w_init)
        bias_list.append(b_init)
        if error == 1:
            w_init = w_init + inputs[i]
            b_init = b_init + error
        elif error == -1:
            w_init = w_init - inputs[i]
            b_init = b_init + error
        elif error == 0:
            w_init = w_init
            b_init = b_init
    return weight_list, bias_list, error_list

#%%

inputs = [[-2,2], [-1,2], [-2,1], [-1,1], [1,1], [1,2], [2,1], [2,2], [1,-1], [1,-2], [2,-1], [2,-2], [-2,-1], [-2,-2], [-1,-2], [-1,-1]]
labels = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

weight, bias, error_list = perceptron(inputs,labels, 10)  

#%%

#Error vs. epochs
plt.plot(np.array(error_list) ** 2)
plt.title("Perceptron sum of squared errors")
plt.xlabel("Iterations")
plt.ylabel("SSE")
plt.savefig('Perceptron_sse.png')
plt.show()


# %%

#Perceptron decision boundary
p1 = np.array(inputs)[:,0]
p2 = np.array(inputs)[:,1]

p1_label_0 = p1[:8]
p2_label_0 = p2[:8]
p1_label_1 = p1[8:]
p2_label_1 = p2[8:]

fig, ax = plt.subplots()
plt.scatter(p1_label_0, p2_label_0, c='blue', label='0')
plt.scatter(p1_label_1, p2_label_1, c='orange', label= '1')
plt.title("Perceptron")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()

ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])

quiver_1 = ax.quiver(0, 0, 0, 0)

def animate(i):
    
    ax.clear()
    
    plt.scatter(p1_label_0, p2_label_0, label='0')
    plt.scatter(p1_label_1, p2_label_1, c='orange', label= '1')
    
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    
    p1_w = float(weight[i][0][0])
    p2_w = float(weight[i][0][1])

    db_slope = (-p1_w)/p2_w
    db_intercept = (-bias[i])/p2_w

    db_1 = db_slope + db_intercept
    
    quiver_1.set_UVC(np.array(p1_w), np.array([p2_w]))
    ax.add_artist(quiver_1)
    ax.axline((1, db_1), slope=db_slope)
    
    plt.title("Perceptron")
    plt.xlabel("$X$")
    plt.ylabel("$Y$")
    plt.legend(loc = 'upper right')
    
    return quiver_1, 

anim = FuncAnimation(fig, animate, frames=160)

anim.save('Perceptron_classification.gif', writer='ffmpeg', fps =10)


# %%


inputs = [[-2,2], [-1,2], [-2,1], [-1,1], [1,1], [1,2], [2,1], [2,2], [1,-1], [1,-2], [2,-1], [2,-2], [-2,-1], [-2,-2], [-1,-2], [-1,-1]]
labels = [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]

weight, bias, error_list = perceptron(inputs,labels, 10)  

#%%

#Error vs. epochs
plt.plot(np.array(error_list) ** 2)
plt.title("Perceptron XOR sum of squared errors")
plt.xlabel("Iterations")
plt.ylabel("SSE")
plt.savefig('Perceptron_sse_XOR.png')
plt.show()

# %%

#Perceptron decision boundary
p1 = np.array(inputs)[:,0]
p2 = np.array(inputs)[:,1]

p1_label_0 = np.concatenate([p1[:4], p1[8:12]])
p2_label_0 = np.concatenate([p2[:4], p2[8:12]])
p1_label_1 = np.concatenate([p1[4:8], p1[12:]])
p2_label_1 = np.concatenate([p2[4:8], p2[12:]])

fig, ax = plt.subplots()
plt.scatter(p1_label_0, p2_label_0, c='blue', label='0')
plt.scatter(p1_label_1, p2_label_1, c='orange', label= '1')
plt.title("Perceptron XOR problem")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()

ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])

quiver_1 = ax.quiver(0, 0, 0, 0)

def animate(i):
    
    ax.clear()
    
    plt.scatter(p1_label_0, p2_label_0, label='0')
    plt.scatter(p1_label_1, p2_label_1, c='orange', label= '1')
    
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    
    p1_w = float(weight[i][0][0])
    p2_w = float(weight[i][0][1])

    db_slope = (-p1_w)/p2_w
    db_intercept = (-bias[i])/p2_w

    db_1 = db_slope + db_intercept
    
    quiver_1.set_UVC(np.array(p1_w), np.array([p2_w]))
    ax.add_artist(quiver_1)
    ax.axline((1, db_1), slope=db_slope)
    
    plt.title("Perceptron XOR problem")
    plt.xlabel("$X$")
    plt.ylabel("$Y$")
    plt.legend(loc = 'upper right')
    
    return quiver_1, 

anim = FuncAnimation(fig, animate, frames=160)

anim.save('Perceptron_classification_XOR.gif', writer='ffmpeg', fps =10)


# %%
