import numpy as np
import matplotlib.pyplot as plt
import math

errors = []

# hypothesis function y = 1/1+e^x1*w1 + b, sigmoid function
def hypothesis(parameters, x_set):
    y = 0
    for i in range(len(parameters)):
        y += parameters[i]*x_set[i]
    
    return 1.0 / (1.0 + math.exp((-1*y)))

# gradient descent
# optimize the function parameters to reduce the loss
# works for linear and logistic regression(SOMEHOW??)
def gradientDescent(parameters, x_set, y_set, learning_rate):

    thetas = list(parameters)

    for i in range(len(parameters)):
        sigma = 0
        for j in range(len(x_set)):
            """
            sigma = 
                [(hypothesis(x1) - y1)*x1]
                        +
                [(hypothesis(x2) - y2)*x2]
                        +
                        ....
                        +
                [(hypothesis(xn) - yn)*xn]
            """
            sigma += (hypothesis(parameters, x_set[j]) - y_set[j])*x_set[j][i] # sum of the errors multiplied by their respective variables
            # !!!remember that the bias has to be accounted for in the number of columns in data and the number of parameters
        """
        thetan = thetan - (alpha/n)*sigma
        """
        thetas[i] = parameters[i] - (learning_rate/len(x_set))*sigma # get the new parameters with the sum
    
    return thetas

# track errors across epochs
# has nothing to do with the optimization in gradient descent
def showError(parameters, x_set, y_set):

    global errors
    total_error = 0
    error = 0
    for i in range(len(x_set)):
        yhat = hypothesis(parameters, x_set[i])
        if(y_set[i] == 1):# log(0) is undefined
            if(yhat == 0):
                yhat = 0.0001
            error = math.log(yhat)*(-1)
        if(y_set[i] == 0):
            if(yhat == 1):
                yhat = 0.9999
            error = math.log(1-yhat)*(-1)
        print( "error %f  hyp  %f  y %f " % (error, yhat,  y_set[i])) 
        total_error += error

    avg_error=total_error/len(x_set)
    errors.append(avg_error)
    return avg_error

         
def scaler(x_set):

    x_set = np.asarray(x_set).T.tolist()

    # start from 1 so as to not try and scale constant for bias
    for i in range(1, len(x_set)):
        minimum = min(x_set[i])
        maximum = max(x_set[i])
        for j in range(len(x_set[i])):
            x_set[i][j] = (x_set[i][j] - minimum) / (maximum - minimum)

    return np.asarray(x_set).T.tolist()

# import dataset
# Note: cannot include the titles of the columns
train_set = np.loadtxt('parkinsons.csv', delimiter=',', skiprows=1, max_rows=160)

# remove y from dataset
x_special = np.delete(train_set, 17, axis=1)

# get the y separated from the rest of the dataset
y_set = np.loadtxt('parkinsons.csv', usecols = 17, delimiter=',', skiprows=1)

# add bias column initialized at 1
x_set = np.insert(x_special, 0, 1, axis=1)

x_set = scaler(x_set)

parameters = [0] * 23

learning_rate = 0.03

epochs = 0

while True:
	oldparams = list(parameters)
	parameters=gradientDescent(parameters, x_set,y_set,learning_rate)	
	error = showError(parameters, x_set, y_set)
	epochs = epochs + 1
	if(oldparams == parameters or epochs > 1000):
		print("final train error: " + str(error))
		print(parameters)
		break

plt.plot(errors)
plt.show()

test_set = np.loadtxt('parkinsons.csv', delimiter=',', skiprows=160)
x_special = np.delete(test_set, 17, axis=1)

y_test = np.loadtxt('parkinsons.csv', usecols = 17, delimiter=',', skiprows=160)
x_test = np.insert(x_special, 0, 1, axis=1)

for i in range(len(x_test)):
    error = showError(parameters, x_test, y_test)
    print("test error: " + str(error))