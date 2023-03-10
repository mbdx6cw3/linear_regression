def gradient_descent(x, y, w_init, b_init, alpha, num_iters):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x (ndarray (m,))  : Data, m examples
      y (ndarray (m,))  : target values
      w_init,b_init (scalar): initial values of model parameters
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent

    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b]
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    import math
    j_history = []
    p_history = []
    b = b_init
    w = w_init

    cost_converge = 1e-12
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            j_history.append(compute_cost(x, y, w, b))
            p_history.append([w, b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 1000) == 0:
            print(f"Iteration {i:4}: Cost {j_history[-1]:0.8e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
        if i > 0:
            if abs(j_history[-1] - j_history[-2]) < cost_converge:
                print(f"Cost converged after{i:4} steps.")
                break
    return w, b, j_history, p_history  # return w and J,w history for graphing


def compute_cost(x, y, w, b):
    """
     Computes the cost for a linear function
    Args:
        x (ndarray (m,)): m data examples
        y (ndarray (m,)): m target values
        w, b (scalar): model parameters
    Returns:
        total_cost: the cost calculated from the sum of square error
        between the m model values, f_wb, and m true values, y.
    """
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = cost / 2 / m

    return total_cost


def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b
     """
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += f_wb - y[i]
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def skl(x, y, tmp_alpha, n):
    from sklearn.linear_model import SGDRegressor
    sgdr = SGDRegressor(max_iter=n, alpha=tmp_alpha, learning_rate='optimal')
    sgdr.fit(x, y)
    w = sgdr.coef_[0]
    b = sgdr.intercept_[0]
    return w, b

def tf(x, y, tmp_alpha, n):
    from keras.layers import Dense
    from keras.models import Sequential
    import numpy as np
    # defined a network with a single linear layer
    linear_layer = Sequential([Dense(1,input_dim=1,activation="linear")])
    # reshape np arrays to tensors for Tf
    X = np.reshape(x, (len(x),1))
    Y = np.reshape(y, (len(y),1))
    linear_layer.compile(loss="mse",optimizer="Adam",metrics=["mae"])
    linear_layer.fit(X[:,0], Y[:,0], epochs=10000,verbose=2)
    w = linear_layer.layers[0].get_weights()[0][0]
    b = linear_layer.layers[0].get_weights()[1]
    print(w,b)
    return w, b
