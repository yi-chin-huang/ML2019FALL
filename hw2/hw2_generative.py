import pandas as pd
import numpy as np
from numpy.linalg import inv
import sys

def load_data(X_train, Y_train, X_test):
	x_train_pd = pd.read_csv(X_train)
	y_train_pd = pd.read_csv(Y_train, header = None)
	x_test_pd = pd.read_csv(X_test)
	x_train =  np.array(x_train_pd)
	y_train = np.array(y_train_pd)
	x_test = np.array(x_test_pd)
	return(x_train, y_train, x_test)

def normalize(x_train, x_test):
    x_all = np.concatenate((x_train, x_test), axis = 0)
    mean = np.mean(x_all, axis = 0)
    std = np.std(x_all, axis = 0)

    index = [0, 1, 3, 4, 5]
    mean_vec = np.zeros(x_all.shape[1])
    std_vec = np.ones(x_all.shape[1])
    mean_vec[index] = mean[index]
    std_vec[index] = std[index]

    x_all_nor = (x_all - mean_vec) / std_vec

    x_train_nor = x_all_nor[0:x_train.shape[0]]
    x_test_nor = x_all_nor[x_train.shape[0]:]

    return x_train_nor, x_test_nor

def train(x, y):
    y = np.reshape(y,(-1,))
    total_num = len(y)
    c0_num = len(np.where(y==1)[0])
    c1_num = len(np.where(y==0)[0])
    c0_idx = np.where(y==1)[0]
    c1_idx = np.where(y==0)[0]
    
    x_c0 = x[c0_idx]
    x_c1 = x[c1_idx]
    
    c0_mu = np.mean(x_c0, axis = 0)
    c1_mu = np.mean(x_c1, axis = 0)
    
    c0_sigma = 1/c0_num * np.dot((x_c0-c0_mu).T, (x_c0-c0_mu))
    c1_sigma = 1/c1_num * np.dot((x_c1-c1_mu).T, (x_c1-c1_mu))
    
    shared_sigma = c0_num/total_num * c0_sigma + c1_num/total_num * c1_sigma
    
    return(c0_num, c1_num, c0_mu, c1_mu, shared_sigma)

def sigmoid(z):
    res = np.clip(1/(1 + np.exp(-z)), 1e-6, 1-1e-6)
    return res

def predict(x_test, c0_num, c1_num, c0_mu, c1_mu, shared_sigma):
    w = np.dot((c0_mu - c1_mu).T, inv(shared_sigma))
    b = -1/2 * np.dot(np.dot(c0_mu.T, inv(shared_sigma)), c0_mu) +1/2 * np.dot(np.dot(c1_mu.T, inv(shared_sigma)), c1_mu) + np.log(c0_num/c1_num)
    z = np.dot(w, x_test.T) + b
    pred = sigmoid(z)
    return pred  
 
def out(res, name):
    f = open(name, 'w')
    print("id,label", file = f)
    for i, y in enumerate(pred_y):
        if y >= 0.5:
            print(i+1, 1, sep = ',', file = f)
        else:
            print(i+1, 0, sep = ',', file = f)

if __name__ == '__main__':
    
	x_train, y_train, x_test = load_data(sys.argv[3], sys.argv[4], sys.argv[5])
	x_train_nor, x_test_nor = normalize(x_train, x_test)
	c0_num, c1_num, c0_mu, c1_mu, shared_sigma = train(x_train_nor, y_train)
	pred_y = predict(x_test_nor, c0_num, c1_num, c0_mu, c1_mu, shared_sigma)
	pred_y = np.around(pred_y)

	out(pred_y, sys.argv[6])

