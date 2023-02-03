#%%
#1. Write a script to initialize and train a network to solve this ”practical” problem. Hint: Implement the learning rule
import numpy as np

inputs = [[1,4], [1,5], [2,4], [2,5], [3,1], [3,2], [4,1], [4,2]]
labels = [0,0,0,0,1,1,1,1]

#%%
def transfer_function(n):
    if n < 0:
        return 0
    else:
        return 1

#%%

def perceptron(inputs, labels, n_iter):
    inputs = np.array(inputs * n_iter)ß
    labels = labels * n_iter
    w_init = np.random.rand(1,2).round(1)
    b_init = 0
    error_list = []
    for i in range(len(inputs)):
        n = np.dot(w_init, inputs[i]) + b_init #n = W*p + b
        a = transfer_function(n) #a = F(n)
        error = labels[i] - a
        error_list.append(error)
        if error == 1:
            w_init = w_init + inputs[i]
            b_init = b_init + error
        elif error == -1:
            w_init = w_init - inputs[i]
            b_init = b_init + error
        elif error == 0:
            w_init = w_init
            b_init = b_init
    return w_init, b_init, error_list

weight, bias, error_list = perceptron(inputs,labels, 2)  

plt.plot(error_list)

#What is the current problem?
#The code does

#what is the end goal of the learning rule?
#For the weight matrix to correctly classify each label
#For error to be 0 for all inputs.
#For n = W*p + b 

# %%
import matplotlib.pyplot as plt


p1 = np.array(inputs)[:,0]
p2 = np.array(inputs)[:,1]
origin = [0,0]

p1_w = float(weight[:,0])
p2_w = float(weight[:,1])

db_slope = (-p1_w)/p2_w
db_intercept = (-bias)/p2_w

fig, ax = plt.subplots()
plt.scatter(p1,p2, c=labels)
ax.axline((0,db_intercept), slope= db_slope)

#%%
#Example: nsolvable problem
new_inputs = [[1,1], [1,4], [4,1], [4,4]]
new_labels = [1,0,0,1]

new_weight, new_bias, new_error_list = perceptron(new_inputs,new_labels,50)

plt.plot(new_error_list)

# %%
p1 = np.array(new_inputs)[:,0]
p2 = np.array(new_inputs)[:,1]
origin = [0,0]

p1_w = float(new_weight[:,0])
p2_w = float(new_weight[:,1])

db_slope = (-p1_w)/p2_w
db_intercept = (-new_bias)/p2_w

fig, ax = plt.subplots()
plt.scatter(p1,p2, c=new_labels)
ax.axline((0,db_intercept), slope= db_slope)
# %%
