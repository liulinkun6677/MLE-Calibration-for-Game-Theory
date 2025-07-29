import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Import dataset
data = pd.read_csv('Preprocessed data.csv')

# Label classification
data['y_label'] = data['x1'].astype(str) + data['x2'].astype(str)

# Define payoff function
def U1(x1, x2, params, data, i):
    a11_1, a11_2, a12_1, a21_1, a21_2, a22_1, a11_0, a12_0, a21_0, a22_0 = params
    if x1 == 1 and x2 == 1:
        return a11_0 + a11_1 * data['aA'][i] + a11_2 * data['aA0'][i]
    elif x1 == 1 and x2 == 2:
        return a12_0 + a12_1 * data['aA'][i]
    elif x1 == 2 and x2 == 1:
        return a21_0 + a21_1 * data['aA'][i] + a21_2 * data['aA0'][i]
    elif x1 == 2 and x2 == 2:
        return a22_0 + a22_1 * data['aA'][i]

def U2(x1, x2, params, data, i):
    b11_1, b11_2, b12_1, b12_2, b21_1, b22_1, b11_0, b12_0, b21_0, b22_0 = params
    if x1 == 1 and x2 == 1:
        return b11_0 + b11_1 * data['aB'][i] + b11_2 * data['aB0'][i]
    elif x1 == 1 and x2 == 2:
        return b12_0 + b12_1 * data['aB'][i] + b12_2 * data['aB0'][i]
    elif x1 == 2 and x2 == 1:
        return b21_0 + b21_1 * data['aB'][i]
    elif x1 == 2 and x2 == 2:
        return b22_0 + b22_1 * data['aB'][i]

# Define softmax function
def softmax2(u1, u2):
    return np.exp(u1) / (np.exp(u1) + np.exp(u2))

# Define likelihood function
def likelihood(params, data):
    a_params = params[:10]
    b_params = params[10:]
    p_values = []
    for i in range(len(data)):
        x1 = data.loc[i, 'x1']
        x2 = data.loc[i, 'x2']

        p_11_12_b = softmax2(U2(1, 1, b_params, data, i), U2(1, 2, b_params, data, i))
        p_12_11_b = 1 - p_11_12_b
        p_21_22_b = softmax2(U2(2, 1, b_params, data, i), U2(2, 2, b_params, data, i))
        p_22_21_b = 1 - p_21_22_b

        p_11_21_a = softmax2(U1(1, 1, a_params, data, i), U1(2, 1, a_params, data, i))
        p_21_11_a = 1 - p_11_21_a
        p_12_21_a = softmax2(U1(1, 2, a_params, data, i), U1(2, 1, a_params, data, i))
        p_21_12_a = 1 - p_12_21_a
        p_12_22_a = softmax2(U1(1, 2, a_params, data, i), U1(2, 2, a_params, data, i))
        p_22_12_a = 1 - p_12_22_a
        p_11_22_a = softmax2(U1(1, 1, a_params, data, i), U1(2, 2, a_params, data, i))
        p_22_11_a = 1 - p_11_22_a

        if x1 == 1 and x2 == 1:
            p = p_11_12_b * p_21_22_b * p_11_21_a + p_11_12_b * p_22_21_b * p_11_22_a
        elif x1 == 1 and x2 == 2:
            p = p_12_11_b * p_21_22_b * p_12_21_a + p_12_11_b * p_22_21_b * p_12_22_a
        elif x1 == 2 and x2 == 1:
            p = p_21_22_b * p_11_12_b * p_21_11_a + p_21_22_b * p_12_11_b * p_21_12_a
        elif x1 == 2 and x2 == 2:
            p = p_22_21_b * p_11_12_b * p_22_11_a + p_22_21_b * p_12_11_b * p_22_12_a

        p_values.append(np.log(p))

    return -np.sum(p_values)

history = {
    'params': [],
    'func_vals': []
}

def callbackFunc(params):
    history['params'].append(params.copy())
    func_val = likelihood(params, data)
    history['func_vals'].append(func_val)
    print(f'Iter {callbackFunc.iter}, likelihood={func_val}')
    print('Optimization iteration:', callbackFunc.iter)
    callbackFunc.iter += 1

params0 = [2] * 20
callbackFunc.iter = 0

res_final = minimize(
    likelihood,
    params0,
    args=(data,),
    method='SLSQP',
    tol=1e-8,
    options={'disp': True, 'maxiter': 10000},
    callback=callbackFunc
)

final_params = res_final.x
a_final = final_params[:10]
b_final = final_params[10:]

print("\nparams：")
print("A：", a_final)
print("B：", b_final)

wrong = 0
for i in range(len(data)):
    x1 = data.loc[i, 'x1']
    x2 = data.loc[i, 'x2']

    p11 = softmax2(U2(1, 1, b_final, data, i), U2(1, 2, b_final, data, i)) * \
          softmax2(U2(2, 1, b_final, data, i), U2(2, 2, b_final, data, i)) * \
          softmax2(U1(1, 1, a_final, data, i), U1(2, 1, a_final, data, i)) + \
          softmax2(U2(1, 1, b_final, data, i), U2(1, 2, b_final, data, i)) * \
          softmax2(U2(2, 2, b_final, data, i), U2(2, 1,b_final, data, i)) * \
          softmax2(U1(1, 1, a_final, data, i), U1(2, 2, a_final, data, i))

    p12 = softmax2(U2(1, 2, b_final, data, i), U2(1, 1, b_final, data, i)) * \
          softmax2(U2(2, 1, b_final, data, i), U2(2, 2, b_final, data, i)) * \
          softmax2(U1(1, 2, a_final, data, i), U1(2, 1, a_final, data, i)) + \
          softmax2(U2(1, 2, b_final, data, i), U2(1, 1, b_final, data, i)) * \
          softmax2(U2(2, 2, b_final, data, i), U2(2, 1, b_final, data, i)) * \
          softmax2(U1(1, 2, a_final, data, i), U1(2, 2, a_final, data, i))

    p21 = softmax2(U2(2, 1, b_final, data, i), U2(2, 2, b_final, data, i)) * \
          softmax2(U2(1, 1, b_final, data, i), U2(1, 2, b_final, data, i)) * \
          softmax2(U1(2, 1, a_final, data, i), U1(1, 1, a_final, data, i)) + \
          softmax2(U2(2, 1, b_final, data, i), U2(2, 2, b_final, data, i)) * \
          softmax2(U2(1, 2, b_final, data, i), U2(1, 1, b_final, data, i)) * \
          softmax2(U1(2, 1, a_final, data, i), U1(1, 2, a_final, data, i))

    p22 = softmax2(U2(2, 2, b_final, data, i), U2(2, 1, b_final, data, i)) * \
          softmax2(U2(1, 1, b_final, data, i), U2(1, 2, b_final, data, i)) * \
          softmax2(U1(2, 2, a_final, data, i), U1(1, 1, a_final, data, i)) + \
          softmax2(U2(2, 2, b_final, data, i), U2(2, 1, b_final, data, i)) * \
          softmax2(U2(1, 2, b_final, data, i), U2(1, 1, b_final, data, i)) * \
          softmax2(U1(2, 2, a_final, data, i), U1(1, 2, a_final, data, i))

    pred = np.argmax([p11, p12, p21, p22])
    true = ['11', '12', '21', '22'].index(str(x1) + str(x2))
    if pred != true:
        wrong += 1

print("y:",wrong)
print("len:",len(data))
acc_all = 1 - wrong / len(data)
print(f"\navg_accuracy: {acc_all:.4f}")