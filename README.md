# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.

## Program:

Program to implement the the Logistic Regression Using Gradient Descent.

Developed by: Shanmuga Vasanth M

RegisterNumber: 212223040191
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data2.txt",delimiter = ',')
x= data[:,[0,1]]
y= data[:,2]
print('Array Value of x:')
x[:5]

print('Array Value of y:')
y[:5]

print('Exam 1-Score graph')
plt.figure()
plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label=' Not Admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()


def sigmoid(z):
  return 1/(1+np.exp(-z))
  
print('Sigmoid function graph: ')
plt.plot()
x_plot = np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot))
plt.show()


def costFunction(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad = np.dot(x.T,h-y)/x.shape[0]
  return j,grad


print('X_train_grad_value: ')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
j,grad = costFunction(theta,x_train,y)
print(j)
print(grad)


print('y_train_grad_value: ')
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

def cost(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j


def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad

print('res.x:')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)


def plotDecisionBoundary(theta,x,y):
  x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
  xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot = np.c_[xx.ravel(),yy.ravel()]
  x_plot = np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot = np.dot(x_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
  plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label='Not Admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel('Exam  1 score')
  plt.ylabel('Exam 2 score')
  plt.legend()
  plt.show()

print('DecisionBoundary-graph for exam score: ')
plotDecisionBoundary(res.x,x,y)

print('Proability value: ')
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)


def predict(theta,x):
  x_train = np.hstack((np.ones((x.shape[0],1)),x))
  prob = sigmoid(np.dot(x_train,theta))
  return (prob >=0.5).astype(int)


print('Prediction value of mean:')
np.mean(predict(res.x,x)==y)
*/
```

## Output:
## Array Value of x:
![238196842-87b2bcb1-9f69-40fd-9da7-789d03b3db7a](https://github.com/user-attachments/assets/6e305306-e57c-4912-9a68-327e57a818b8)

## Array Value of y:
![238196875-48635643-f05c-48cb-8b21-d5b0d7e60bf5](https://github.com/user-attachments/assets/72ebf3ee-aefa-4b6b-9e10-fd2ca20b3303)

## Score graph:
![238196912-03841143-dec0-4bdf-beb0-08894db9b552](https://github.com/user-attachments/assets/67c06098-6e96-4c1e-8e19-96e0b6c4177d)

## Sigmoid function graph:
![238196954-8a1ab219-1661-4a15-ba5a-222974a4a948](https://github.com/user-attachments/assets/dd786eb5-39c4-4b24-b14e-e6ed892e4f38)

## X_train_grad value:
![238197005-bfedd65f-0a94-4042-b075-d3ed73047248](https://github.com/user-attachments/assets/74d4abbf-94de-4bb7-ae34-439b2f5bcb6c)

## Y_train_grad value:
![238197045-ff2cbc3f-c96d-4044-96a0-b60f72d18c67](https://github.com/user-attachments/assets/6fb10dc9-4f31-4809-b0e9-f72058065f12)

## res.x:
![238197137-50d9a6fa-887a-4950-b765-1111b31db73e](https://github.com/user-attachments/assets/6d7d457c-b4d7-4dba-be06-ad4bc8664389)

## Decision boundary:
![image](https://github.com/user-attachments/assets/459a6dbf-3a47-4717-84da-9af1ffb18f1d)

## Proability value:
![image](https://github.com/user-attachments/assets/d7647877-c079-4ea2-82c2-a37a61558bfe)

## Prediction value of mean:
![image](https://github.com/user-attachments/assets/f03053da-41cf-4e07-93be-5fdd3ddb7aa4)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

