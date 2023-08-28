#%%

from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt 
import numpy as np

inputs = [[-2,2], [-1,2], [-2,1], [-1,1], [1,1], [1,2], [2,1], [2,2], [1,-1], [1,-2], [2,-1], [2,-2], [-2,-1], [-2,-2], [-1,-2], [-1,-1]]
labels = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

model = Perceptron(tol=1e-3, random_state=123).fit(inputs, labels)
model.score(inputs, labels)

#Perceptron plot
p1 = np.array(inputs)[:,0]
p2 = np.array(inputs)[:,1]

p1_label_0 = p1[:8]
p2_label_0 = p2[:8]
p1_label_1 = p1[8:]
p2_label_1 = p2[8:]

fig, ax = plt.subplots()
plt.scatter(p1_label_0, p2_label_0, label='0')
plt.scatter(p1_label_1, p2_label_1, c='orange', label= '1')
plt.title("Perceptron Sklearn")
plt.xlabel("$X$")
plt.ylabel("$Y$")
plt.legend()
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])

p1_w = model.coef_[0][0]
p2_w = model.coef_[0][1]
db_slope = (-p1_w)/p2_w
db_intercept = (model.intercept_[0])/p2_w
db_1 = db_slope + db_intercept

plt.quiver(np.array(p1_w), np.array([p2_w]))
ax.axline((1, db_1), slope=db_slope)
plt.show()


# %%
