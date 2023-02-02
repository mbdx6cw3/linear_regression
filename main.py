# This is a machine learning program
import load_training
import regression
import matplotlib.pyplot as plt
file_name = "net_liquidity.xlsx"
x_train, y_train = load_training.load_from_xl(file_name)
plt.scatter(x_train,y_train,marker="x",c="r")
plt.ylabel("S&P500 Price ($)")
plt.xlabel("Net Liquidity ($tn)")
plt.title("Net Liquidity Model")
plt.show()

#x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])   #features
#y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

# initialise parameters
w_init = 0.0
b_init = 0.0
iterations = 100000

# set the learning rate
tmp_alpha = 0.02

# TODO: implement Z-score normalisation
#mu_x = np.mean(x_train, axis=0)
#sigma_x = np.std(x_train, axis=0)
#x_train = (x_train - mu_x) / sigma_x

w_final, b_final, J_hist, p_hist = regression.gradient_descent(
    x_train, y_train, w_init, b_init, tmp_alpha, iterations)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

plt.scatter(x_train,y_train,marker="x",c="r")
plt.plot(x_train,w_final*x_train+b_final)
plt.ylabel("S&P500 Price ($)")
plt.xlabel("Net Liquidity ($tn)")
plt.title("Net Liquidity Model")
plt.show()

