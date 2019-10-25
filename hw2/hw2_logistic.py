import pandas as pd
import numpy as np
import sys

def load_data(X_train, Y_train, X_test):
    x_train_pd = pd.read_csv(X_train)
    y_train_pd = pd.read_csv(Y_train, header = None)
    x_test_pd = pd.read_csv(X_test)

    x_train =  add_constant(np.array(x_train_pd))
    x_test = add_constant(np.array(x_test_pd))

    x_train =  np.array(x_train)
    y_train = np.array(y_train_pd)
    x_test = np.array(x_test)
    return(x_train, y_train, x_test)

def add_constant(x):
    return np.c_[x, np.ones(x.shape[0])]

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

def add_features(train_data, test_data):
    feat = [0,1,3,4,5]
    train_data = np.concatenate((train_data, train_data[:, feat]**2, train_data[:, feat]**3, np.exp(train_data[:, feat])), axis = 1)
    test_data = np.concatenate((test_data, test_data[:, feat]**2, test_data[:, feat]**3, np.exp(test_data[:, feat])), axis = 1)
    return(train_data, test_data)

def sigmoid(z):
    res = np.clip(1/(1 + np.exp(-z)), 1e-6, 1-1e-6)
    return res

def train(x, y, l, it):
    xss = x
    yss = y.reshape(-1,)
    num_data, num_feature = xss.shape
    ws = np.zeros(num_feature)
    lr = l
    prev_gra = np.zeros(num_feature)
    loss = 0
    for j in range(10):
        for i in range(it):
            z = np.dot(xss,ws)
            predict = sigmoid(z)
            diffs = yss - predict
            grad = -np.dot(diffs,xss) * 2
            prev_gra += grad**2
            ada = np.sqrt(prev_gra)
            ws -= lr * grad/ada

        pred_y = [1 if i >= 0.5 else 0 for i in predict]        
        result = (np.reshape(y_train,(-1,)) == pred_y)
        
    return(ws)

def sigmoid(z):
    res = np.clip(1/(1 + np.exp(-z)), 1e-6, 1-1e-6)
    return res

def predict(x_test, c0_num, c1_num, c0_mu, c1_mu, shared_sigma):
    w = np.dot((c0_mu - c1_mu).T, inv(shared_sigma))
    b = -1/2 * np.dot(np.dot(c0_mu.T, inv(shared_sigma)), c0_mu) +1/2 * np.dot(np.dot(c1_mu.T, inv(shared_sigma)), c1_mu) + np.log(c0_num/c1_num)
    z = np.dot(w, x_test.T) + b
    pred = sigmoid(z)
    return pred

def my_predict(test_x, ws, name):    
    sample = pd.read_csv('./data/sample_submission.csv')
    pred_y = np.dot(test_x,ws)
    # make csv
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
    x_train_add, x_test_add = add_features(x_train_nor, x_test_nor)
    ws = train(x_train_add, y_train, 0.01, 1000)
    my_predict(x_test_add, ws, sys.argv[6])

