
# coding: utf-8

# In[164]:


import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data set
data = np.loadtxt('C:/Users/Mina Alpu/Desktop/polynome.data')
# Separate the input from the output
X = data [:, 0]
Y = data [:, 1]
N = len(X)

def visualize(w, X, Y):
    
    plt.plot(X,Y, 'r.')
    x = np.linspace(0., 1., 100)
    y = np.polyval(w, x)
    plt.plot(x, y, 'g-')
    plt.title('Polynomial regression with order' + str(len(w)-1))
    plt.show()
    
#degrees ranging from 1 to 20
def cal_error():
    for i in range(1, 21):
        w = np.polyfit(X, Y, i)
        visualize(w, X, Y)
        y_pred = np.polyval(w,X)
        training_error = mean_squared_error(y_pred, Y)
        print("training error: ", training_error)            
cal_error()


#split the dataset as a training and test
x_split = np.hsplit(X, [11])
y_split = np.hsplit(Y, [11])

x_split[0] #for X train set
x_split[1] #for X test set
y_split[0] #for Y train set
y_split[1] #for Y test set


def test_error():
    for i in range(1, 21):
        w = np.polyfit(x_split[0], y_split[0], i)
        visualize(w, x_split[0], y_split[0])
        y_predicted = np.polyval(w, x_split[1])
        test_error = mean_squared_error(y_predicted, y_split[1])
        print("test error: ", test_error) #from cross validation
test_error()

x_split = np.hsplit(X, 2) #for k=2
y_split = np.hsplit(Y, 2)

x_test = x_split[1]

x_train = np.hstack ((x_split[i] for i in range(2) if not i==1))
print("if fold 1 is not observed, X training data: ", x_train)
print(x_test)

y_train = np.hstack ((y_split[i] for i in range(2) if not i==1))
print("if fold 1 is not observed, Y training data", y_train)

x_train1 = np.hstack ((x_split[i] for i in range(2) if not i==0))
print("if first fold is not observed, X training data:  ", x_train1)

y_train1 = np.hstack ((y_split[i] for i in range(2) if not i==0))
print("if first fold is not observed, Y training data:  ", y_train1)

