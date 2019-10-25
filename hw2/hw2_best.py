import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import GradientBoostingClassifier
import pickle

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

def add_features(x_train_nor, x_test_nor):
    feat = [0,1,3,4,5]
    x_train = np.concatenate((x_train_nor, x_train_nor[:, feat]**2, np.exp(x_train_nor[:, feat])), axis = 1)
    x_test = np.concatenate((x_test_nor, x_test_nor[:, feat]**2, np.exp(x_test_nor[:, feat])), axis = 1)
    return x_train, x_test
  
def out(res, name):
    f = open(name, 'w')
    print("id,label", file = f)
    for i, r in enumerate(res):
        print(i+1, r, sep = ',', file = f)

def train(x_train, y_train):
    learning_rate = 0.1
    max_features = 0.9
    max_depth = 3
    n_estimators = 700
    gb_clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_features=max_features, max_depth=max_depth, random_state=0)
    gb_clf.fit(x_train, y_train)
    return gb_clf

if __name__ == "__main__": 
    x_train, y_train, x_test = load_data(sys.argv[3], sys.argv[4], sys.argv[5])
    x_train_nor, x_test_nor = normalize(x_train, x_test)
    x_train_add, x_test_add = add_features(x_train_nor, x_test_nor)
    # gb_clf = train(x_train_add, np.reshape(y_train,(-1,)))
    # with open('model/gdb_clf.pickle', 'wb') as f:
    #     pickle.dump(gb_clf, f)
    with open('model/gdb_clf.pickle', 'rb') as f:
        gb_clf = pickle.load(f)
    pred_y = gb_clf.predict(x_test_add)
    out(pred_y, sys.argv[6])

    