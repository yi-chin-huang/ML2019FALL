import numpy as np
import pandas as pd
import sys

def readdata(data):
    for col in list(data.columns[2:]):
        data[col] = data[col].astype(str).map(lambda x: x.rstrip('x*#A'))
    data = data.values
    data = np.delete(data, [0,1], 1)
    data[ data == 'NR'] = 0
    data[ data == ''] = 0
    data[ data == 'nan'] = 0
    data = data.astype(np.float)

    return data

def extract(data):
    N = data.shape[0] // 18
    temp = data[:18, :]
    
    for i in range(1, N):
        temp = np.hstack((temp, data[i*18: i*18+18, :]))
    return temp

def get_mean_std(data1, data2):
    pm25 = np.concatenate((data1[9,:], data2[9,:]), axis=None)
    pm25 = [i for i in pm25 if i > 2 and i <= 100]
    pm25_mean = np.mean(pm25)
    pm25_std = np.std(pm25)
    return(pm25_mean,pm25_std)

def parse2test2(data, mean_test, std_test, a):
    x = []
    
    num_data = data.shape[1]//9
    for i in range(num_data):
        x_tmp = data[:,i*9:(i+1)*9]
        pm25 = x_tmp[9,:]
        for i, v in enumerate(pm25):
            if v <= mean_test - a*std_test or v >= mean_test + a*std_test:
                pm25[i] = mean_test
        x_tmp[9,:] = pm25
        x.append(x_tmp.reshape(-1,))
    x = np.array(x)
    return x

def add_constant(x):
    return(np.c_[x, np.ones(x.shape[0])])

def my_predict(test_x, ws, out):    
    sample = pd.read_csv('./data/sample_submission.csv')
    pred_y = np.dot(test_x,ws)
    # make csv
    for i, y in enumerate(pred_y):
        sample.at[i,'value'] = y
    sample.value = np.where(sample.value < 0, 0,sample.value) #remove anomalies
    sample.to_csv(out, index=False)

if __name__ == "__main__":
	test_name, out_name = sys.argv[1], sys.argv[2]
	test_pd = pd.read_csv(test_name)
	test_pd1 = readdata(test_pd)
	test_pd2 = extract(test_pd1)
	mean_test, std_test = np.mean(test_pd2[9,:]),  np.std(test_pd2[9,:])
	x_test = parse2test2(test_pd2, mean_test, std_test, 3.8)
	x_test = add_constant(x_test)
	w = np.load('./data/weight.npy')
	my_predict(x_test, w, out_name)

