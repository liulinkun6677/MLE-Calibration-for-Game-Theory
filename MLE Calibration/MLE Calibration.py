import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Import dataset
data = pd.read_csv('Preprocessed data.csv')

# Define payoff function
def U1(x1, x2, a11_0, a12_0,a21_0,a22_0,a11_1, a11_2,a12_1, a21_1, a21_2, a22_1,i):
    if x1 == 1 and x2 == 1:
        return a11_0+a11_1 * data['aA'][i]  + a11_2 * data['aA0'][i]
    elif x1 == 1 and x2 == 2:
        return a12_0+a12_1 * data['aA'][i]
    elif x1 == 2 and x2 == 1:
        return a21_0+a21_1 * data['aA'][i] + a21_2 * data['aA0'][i]
    elif x1 == 2 and x2 == 2:
        return a22_0+a22_1 * data['aA'][i]
def U2(x1, x2, b11_0, b12_0,b21_0,b22_0,b11_1, b11_2, b12_1, b12_2, b21_1, b22_1,i):
    if x1 == 1 and x2 == 1:
        return b11_0+b11_1 * data['aB'][i] + b11_2 * data['aB0'][i]
    elif x1 == 1 and x2 == 2:
        return b12_0+b12_1 * data['aB'][i] + b12_2 * data['aB0'][i]
    elif x1 == 2 and x2 == 1:
        return b21_0+b21_1 * data['aB'][i]
    elif x1 == 2 and x2 == 2:
        return b22_0+b22_1 * data['aB'][i]

# Define likelihood function
def likelihood(params):
    a11_1, a11_2,a12_1, a21_1, a21_2, a22_1, b11_1, b11_2, b12_1, b12_2, b21_1, b22_1,a11_0, a12_0,a21_0,a22_0,b11_0, b12_0,b21_0,b22_0 = params
    p_values = []
    for i in range(len(data)):
        x1 = data.loc[i, 'x1']
        x2 = data.loc[i, 'x2']

        p_11_12_b = (np.exp(U2(1, 1,b11_0, b12_0,b21_0,b22_0, b11_1, b11_2, b12_1, b12_2, b21_1, b22_1,i)))/ (
            np.exp(U2(1, 1, b11_0, b12_0,b21_0,b22_0,b11_1, b11_2, b12_1, b12_2, b21_1, b22_1,i))+
            np.exp(U2(1, 2, b11_0, b12_0,b21_0,b22_0,b11_1, b11_2, b12_1, b12_2, b21_1, b22_1,i)))

        p_12_11_b = 1 - p_11_12_b

        p_21_22_b = (np.exp(U2(2, 1, b11_0, b12_0,b21_0,b22_0,b11_1, b11_2, b12_1, b12_2, b21_1, b22_1,i)))/ (
            np.exp(U2(2, 1, b11_0, b12_0,b21_0,b22_0,b11_1, b11_2, b12_1, b12_2, b21_1, b22_1,i))+
            np.exp(U2(2, 2, b11_0, b12_0,b21_0,b22_0,b11_1, b11_2, b12_1, b12_2, b21_1, b22_1,i)))

        p_22_21_b = 1 - p_21_22_b

        p_11_21_a = (np.exp(U1(1, 1, a11_0, a12_0,a21_0,a22_0,a11_1, a11_2,a12_1, a21_1, a21_2, a22_1,i)))/ (
            np.exp(U1(1, 1, a11_0, a12_0,a21_0,a22_0,a11_1, a11_2,a12_1, a21_1, a21_2, a22_1,i))+
            np.exp(U1(2, 1, a11_0, a12_0,a21_0,a22_0,a11_1, a11_2,a12_1, a21_1, a21_2, a22_1,i)))

        p_21_11_a = 1 - p_11_21_a

        p_12_21_a = (np.exp(U1(1, 2, a11_0, a12_0,a21_0,a22_0,a11_1, a11_2,a12_1, a21_1, a21_2, a22_1,i)))/ (
            np.exp(U1(1, 2, a11_0, a12_0,a21_0,a22_0,a11_1, a11_2,a12_1, a21_1, a21_2, a22_1,i))+
            np.exp(U1(2, 1, a11_0, a12_0,a21_0,a22_0,a11_1, a11_2,a12_1, a21_1, a21_2, a22_1,i)))

        p_21_12_a = 1 - p_12_21_a

        p_12_22_a = (np.exp(U1(1, 2, a11_0, a12_0,a21_0,a22_0,a11_1, a11_2,a12_1, a21_1, a21_2, a22_1,i)))/ (
            np.exp(U1(1, 2, a11_0, a12_0,a21_0,a22_0,a11_1, a11_2,a12_1, a21_1, a21_2, a22_1,i))+
            np.exp(U1(2, 2, a11_0, a12_0,a21_0,a22_0,a11_1, a11_2,a12_1, a21_1, a21_2, a22_1,i)))

        p_22_12_a = 1 - p_12_22_a

        p_11_22_a = (np.exp(U1(1, 1, a11_0, a12_0,a21_0,a22_0,a11_1, a11_2,a12_1, a21_1, a21_2, a22_1,i)))/ (
            np.exp(U1(1, 1, a11_0, a12_0,a21_0,a22_0,a11_1, a11_2,a12_1, a21_1, a21_2, a22_1,i))+
            np.exp(U1(2, 2, a11_0, a12_0,a21_0,a22_0,a11_1, a11_2,a12_1, a21_1, a21_2, a22_1,i)))

        p_22_11_a = 1 - p_11_22_a

        if x1 == 1 and x2 == 1:
            p = p_11_12_b*p_21_22_b*p_11_21_a + p_11_12_b*p_22_21_b*p_11_22_a

        elif x1 == 1 and x2 == 2:
            p = p_12_11_b*p_21_22_b*p_12_21_a + p_12_11_b*p_22_21_b*p_12_22_a

        elif x1 == 2 and x2 == 1:
            p = p_21_22_b*p_11_12_b*p_21_11_a + p_21_22_b*p_12_11_b*p_21_12_a

        elif x1 == 2 and x2 == 2:
            p = p_22_21_b*p_11_12_b*p_22_11_a + p_22_21_b*p_12_11_b*p_22_12_a

        p_values.append(np.log(p))

    return -np.sum(p_values)

# Minimize the likelihood function
params0 = [2]*20

# Define the convergence tolerance
tol = 1e-8

# Setting parameters and optimizer details
options = {'disp': True, 'maxiter': 10000}

# Define callback function
def callbackFunc(params):
    print('Optimization iteration:', callbackFunc.iter)
    callbackFunc.iter += 1

callbackFunc.iter = 0

res = minimize(likelihood, params0, method='SLSQP', tol=tol, options=options, callback=callbackFunc)

# Output results
print("results：", res.x)




