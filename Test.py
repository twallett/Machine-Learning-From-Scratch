#%% [markdown]

#### $\alpha_k = - \frac{\nabla f(x_k)^T \cdot p_k^T} {p_k^T \cdot  H_f(x_k) \cdot p_k} $

#### $x_{k+1} = x_k - \alpha \nabla f(x_{k-1})$

# * $\alpha$: The learning rate.
# * $x_k$: The initial condition.
# * $\nabla f(x_k)$: The gradient.
# * $H_f(x_k)$: The Hessian.

## $\text{hardlim}(n) = \begin{cases} 0 & \text{if } n < 0 \\ 1 & \text{if } n \geq 0 \end{cases}$


# $s^{m} = F^{m} \cdot (n^{m}) \cdot (W^{m+1^{T}}) \cdot s^{m+1}$

#%% [markdown]

# $f(x) = 1 + sin(\frac{\pi}{4}x)$

# %%
