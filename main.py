# This is a machine learning program
import load_training
import matplotlib.pyplot as plt
import regression
import normalise
import time

# TODO: request user input to determine input data type (e.g. .txt file, .xlsx file)
# TODO: adapt code so that it can deal with required number of features
# TODO: add exceptions for user input here
file_name = input("Enter the path to the file containing the dataset: ")
n_feature = int(input("Enter the number of features to use for training: "))
reg_tool = input("Use scikit-learn (Y/N): ")
if reg_tool == "Y":
    reg_tool = True
else:
    reg_tool = False

print(reg_tool)

x_train, y_train = load_training.load_from_xl(file_name, n_feature)

# plot dataset
plt.scatter(x_train, y_train, marker="x", c="r")
plt.ylabel("y")
plt.xlabel("x")
plt.title("Linear Regression")
plt.show()

# initialise parameters
w_init = 0.0
b_init = 0.0
max_iter = 100000  # set the maximum number of iterations in gradient descent
tmp_alpha = 0.1  # set the learning rate
w_norm = 0.0
b_norm = 0.0

# normalise features
tic = time.perf_counter()
if reg_tool:
    # reshape to 2D array TODO: may need to also do this for multiple linear regression
    x_train = x_train.reshape(-1, 1)
    x_norm = normalise.skl(x_train)
    w_norm, b_norm = regression.skl(x_norm, y_train, max_iter)
else:
    x_norm, mu_x, sigma_x = normalise.z_score(x_train)
    w_norm, b_norm, J_hist, p_hist = regression.gradient_descent(
        x_norm, y_train, w_init, b_init, tmp_alpha, max_iter)
toc = time.perf_counter()
print(f"Time taken to do linear regression computation: {toc - tic:0.4f} seconds")

print(f"Normalized (w,b) found by gradient descent: ({w_norm:8.4f}, {b_norm:8.4f})")

plt.scatter(x_norm, y_train, marker="x", c="r")
plt.plot(x_norm, w_norm*x_norm+b_norm)
plt.ylabel("y")
plt.xlabel("x")
plt.title("Linear Regression")
plt.show()
