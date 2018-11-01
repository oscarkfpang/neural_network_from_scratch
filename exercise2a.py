import numpy as np

def mean_square_loss(Y, Y_hat):
    sq_error = (y - y_hat)**2
    sum_of_sq = np.sum(sq_error)
    m = y.size
    return sum_of_sq / m

def cross_entropy_loss(Y, Y_hat):
    return None

# make sure our output is the same
np.random.seed(0)

# generate some random values of y and y^
y = np.random.rand(3)
y_hat = np.random.rand(3)

# run the mean_square_loss and cross_entropy_loss
print("MSE loss = ",mean_square_loss(y, y_hat))
print("Cross Entroy Error loss = ",cross_entropy_loss(y, y_hat))
