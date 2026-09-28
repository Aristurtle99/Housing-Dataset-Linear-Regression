import numpy as np
import matplotlib.pyplot as plt
import math, copy
import pandas as pd
import matplotlib.pyplot as plt
from plt_gradients import plt_gradients

df = pd.read_csv("/Users/wildan/Downloads/Machine_Learning/ml_/data/housing.csv")
x_train = df["median_income"].values
x_train = (x_train - np.mean(x_train)) / np.std(x_train)
y_train = df["median_house_value"].values
y_train = (y_train - np.mean(y_train)) / np.std(y_train)
#y_train = np.array([300.0,400.0])#target value


print(df.head())

def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = x[i]*w + b
        cost = (y[i] - f_wb)**2
        cost_sum = cost_sum + (y[i] - f_wb)**2
    total_cost = 1/(2*m) * cost_sum
    return total_cost



def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw =0
    dj_db = 0

    for i in range(m):
        f_wb = x[i]*w + b
        dj_dw_i = (-y[i] + f_wb) * x[i]
        dj_db_i = -y[i] + f_wb
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m


    return dj_dw,dj_db

def gradient_descent(x,y,w_in,b_in,alpha,num_iters,cost_fn,grad_fn):
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    for i in range(num_iters):
        dj_dw,dj_db = grad_fn(x,y,w,b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        if i<1000000:
            J_history.append(cost_fn(x,y,w,b))
            p_history.append([w,b])

        if i%math.ceil(num_iters/10)==0:
          
          print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w,b,J_history,p_history




w_init=0
b_init=0
itr = 2000
t_alpha = 0.05
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, t_alpha, 
                                                    itr, compute_cost, compute_gradient)

print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
#plt_gradients(x_train,y_train, compute_cost(x_train,y_train), compute_gradient)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# -------------------------
# 1. Cost vs iteration
# -------------------------

ax1.plot(J_hist)

ax1.set_title("Cost vs. Iteration")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Cost")


# -------------------------
# 2. Scatter + regression line
# -------------------------

indices = np.random.choice(len(x_train), 40, replace=False)

x_line = np.linspace(x_train.min(), x_train.max(), 100)
y_line = w_final * x_line + b_final

ax2.scatter(
    x_train[indices],
    y_train[indices],
    marker='x',c='r',
    label='Actual Values'
)

ax2.plot(
    x_line,
    y_line,
    label='Gradient Descent'
)

ax2.set_title("Linear Regression")
ax2.set_xlabel("Standardized Median Income")
ax2.set_ylabel("Standardized Median House Value")
ax2.legend()

plt.show()