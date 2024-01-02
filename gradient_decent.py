import numpy as np

def gradient_decent(x, y):
    w_curr = b_curr = 0
    iterations = 1000
    learning_rate = -0.01
    m = len(x)
    for i in range(iterations):
        #computing the y predicted value
        y_predicted = w_curr * x + b_curr

        #squarred error cost function
        cost = (1/(2*m)) * sum([val**2 for val in (y - y_predicted)])

        #computing the derivatives 
        w_deriv = (1/m) * sum(x*(y - y_predicted))
        b_deriv = (1/m) * sum(y-y_predicted)

        #simulataneously updating them
        w_curr = w_curr - learning_rate * w_deriv
        b_curr = b_curr - learning_rate * b_deriv

        #printing all the values we have
        print(f"w: {w_curr}, b: {b_curr}, iteration: {i}, cost: {cost}")

x = np.array([1, 2, 3, 4 ,5])
y = np.array([5, 6, 7, 8, 9])

gradient_decent(x, y)