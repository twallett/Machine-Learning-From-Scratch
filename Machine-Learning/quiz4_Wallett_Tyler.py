import matplotlib.pyplot as plt
import numpy as np

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

# -------------------------------------------------
# Adaline network and use LMS

# wnew =  wold + 2*alpha*error*search direction

def adaline(inputs, labels, alpha, n_iter):
    inputs = np.concatenate([inputs] * n_iter, axis=1)
    labels = np.concatenate([labels] * n_iter, axis=1)
    w_init = np.identity(2)
    b_init = np.array([1, 1]).reshape((2, 1))
    error_l = []
    for i in range(1, inputs.shape[1] + 1):
        error = labels[:, i - 1:i] - (w_init @ inputs[:, i - 1:i] + b_init)
        w_init = w_init + 2 * alpha * error @ inputs[:, i - 1:i].T
        b_init = b_init + 2 * alpha * error
        error_l.append(error)
        if error[0] == 0 and error[1] == 0:
            break
        else:
            continue
    return w_init, b_init, error_l


w, b, e_list = adaline(p, t, 0.04, 10)

# -------------------------------------------------
# Calculate the sum square error and plot it

e_list_sq = np.array(e_list) ** 2

plt.plot(e_list_sq[:, 0])
plt.plot(e_list_sq[:, 1])
plt.legend(['X1', 'X2'])
plt.ylabel("Sum of squared errors")
plt.xlabel("Iterations")
plt.title("Sum of squared errors plot")
plt.show()

# -------------------------------------------------
# Plot the Decision Boundaries and the patterns.

fig, ax = plt.subplots()
plt.scatter(p[0, :2], p[1, :2], marker='s')
plt.scatter(p[0, 2:4], p[1, 2:4], marker='o')
plt.scatter(p[0, 4:6], p[1, 4:6], marker='x')
plt.scatter(p[0, 6:8], p[1, 6:8], marker='P')
plt.ylabel('X2')
plt.xlabel('X1')
plt.title('X2 vs. X1')

w_1, w_2 = w[0], w[1]
b_1, b_2 = b[0][0], b[1][0]

db1_slope = (-w_1[0]) / (w_1[1])
db2_slope = (-w_2[0]) / (w_2[1])

db1_int = (-b_1 / w_1[1])
db2_int = (-b_2 / w_2[1])

plt.quiver(0, 0, w_1[0], w_1[1], units='xy', scale=1)
ax.axline((0, db1_int), slope=db1_slope)

plt.quiver(0, 0, w_2[0], w_2[1], units='xy', scale=1)
ax.axline((0, db2_int), slope=db2_slope)

plt.show()

# -------------------------------------------------
# Compare the results of part iv. with decision boundary that you will get with perceprtron learning rule. Explain the differences?

def transfer_function(n):
    if n[0] < 0 and n[1] < 0:
        return np.array([0, 0]).reshape(2, 1)
    if n[0] < 0 and n[1] >= 0:
        return np.array([0, 1]).reshape(2, 1)
    if n[0] >= 0 and n[1] < 0:
        return np.array([1, 0]).reshape(2, 1)
    if n[0] >= 0 and n[1] >= 0:
        return np.array([1, 1]).reshape(2, 1)


# %%


t_new = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 0, 1, 1, 0, 0, 1, 1]])


def perceptron(inputs, labels, n_iter):
    inputs = np.concatenate([inputs] * n_iter, axis=1)
    labels = np.concatenate([labels] * n_iter, axis=1)
    w_init = np.identity(2)
    b_init = np.array([1, 1]).reshape((2, 1))
    error_l = []
    for i in range(1, inputs.shape[1] + 1):
        n = (w_init @ inputs[:, i - 1:i] + b_init)
        a = transfer_function(n)
        error = labels[:, i - 1:i] - a
        w_init = w_init + error @ inputs[:, i - 1:i].T
        b_init = b_init + error
        error_l.append(error)
        # if error[0] == 0 and error[1] == 0:
        #     break
        # else:
        #     continue
    return w_init, b_init, error_l


w_p, b_p, e_list_p = perceptron(p, t_new, 20)

# %%

e_list_sq_p = np.array(e_list_p) ** 2

plt.plot(e_list_sq_p[:, 0])
plt.plot(e_list_sq_p[:, 1])
plt.legend(['X1', 'X2'])
plt.ylabel("Sum of squared errors")
plt.xlabel("Iterations")
plt.title("Sum of squared errors plot")
plt.show()

# %%

fig, ax = plt.subplots()
plt.scatter(p[0, :2], p[1, :2], marker='s')
plt.scatter(p[0, 2:4], p[1, 2:4], marker='o')
plt.scatter(p[0, 4:6], p[1, 4:6], marker='x')
plt.scatter(p[0, 6:8], p[1, 6:8], marker='P')
plt.ylabel('X2')
plt.xlabel('X1')
plt.title('X2 vs. X1')

w_p1, w_p2 = w_p[0], w_p[1]
b_p1, b_p2 = b_p[0][0], b_p[1][0]

db2_slope_p = (-w_p2[0]) / (w_p2[1])

db1_int_p = (-b_p1 / w_p1[0])
db2_int_p = (-b_p2 / w_p2[1])

plt.quiver(0, 0, w_p1[0], w_p1[1], units='xy', scale=1)
ax.axvline(x=db1_int_p)

plt.quiver(0, 0, w_p2[0], w_p2[1], units='xy', scale=1)
ax.axline((0, db2_int_p), slope=db2_slope_p)

plt.show()

print(f"One big difference between these two network architectures are the transfer functions that are used. For the Adaline network the purelin function outputs a continuous number, hence its sum square errors continue to converge to zero in a continuous fashion. On the other hand, the perceptron uses the hardlim transfer function which outputs discrete numbers 0 or 1, which are evidenced in the discrete steps in the sum square errors. ")
print(f"In terms of the decision boundaries, the Adaline network has a decision boundary that looks to minimize the least sum of squares, which the perceptron does not care about.")