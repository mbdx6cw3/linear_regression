# This is a machine learning program
import load_training
import matplotlib.pyplot as plt
import regression
import normalise

# TODO: request user input to determine input data type (e.g. .txt file, .xlsx file)
# TODO: adapt code so that it only includes required number of features

file_name = input("Enter the path to the file containing the dataset: ")
n_feature = int(input("Enter the number of features to use for training: "))
x_train, y_train = load_training.load_from_xl(file_name, n_feature)
plt.scatter(x_train, y_train, marker="x", c="r")
plt.ylabel("y")
plt.xlabel("x")
plt.title("Linear Regression")
plt.show()

# x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])   #features
# y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

# initialise parameters
w_init = 0.0
b_init = 0.0
iterations = 10000

# set the learning rate
tmp_alpha = 0.1

# TODO: request user input to request whether to calculate by scikit learn

# normalise data
x_norm, mu_x, sigma_x = normalise.z_score(x_train)

w_norm, b_norm, J_hist, p_hist = regression.gradient_descent(
    x_norm, y_train, w_init, b_init, tmp_alpha, iterations)
print(f"(w,b) found by gradient descent: ({w_norm:8.4f},{b_norm:8.4f})")

plt.scatter(x_norm, y_train, marker="x", c="r")
plt.plot(x_norm, w_norm*x_norm+b_norm)
plt.ylabel("S&P500 Price ($)")
plt.xlabel("Net Liquidity ($tn)")
plt.title("Net Liquidity Model")
plt.show()
