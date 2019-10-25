import numpy as np
import pandas as pd

def readdata(data):
    
    # 把有些數字後面的奇怪符號刪除
    for col in list(data.columns[2:]):
        data[col] = data[col].astype(str).map(lambda x: x.rstrip('x*#A'))
    data = data.values
    
    # 刪除欄位名稱及日期
    data = np.delete(data, [0,1], 1)

    # 特殊值補 0
    data[ data == 'NR'] = 0
    data[ data == ''] = 0
    data[ data == 'nan'] = 0
    data = data.astype(np.float)

    return data

def extract(data):
    N = data.shape[0] // 18
    temp = data[:18, :]
    
    # Shape 會變成 (x, 18) x = 取多少hours
    for i in range(1, N):
        temp = np.hstack((temp, data[i*18: i*18+18, :]))
    return temp

def valid2(x, y, mean, std, a): #去掉 mean +- a*std 外的outlier
    lower = mean - a * std
    upper = mean + a * std
    if y <= lower or y >= upper:
        return False
    for i in range(9):
        if x[9,i] <= lower or x[9,i] >= upper:
            return False
    return True

def parse2train2(data, pm25mean, pm25std, a):
    x = []
    y = []
    # 用前面9筆資料預測下一筆PM2.5 所以需要-9
    total_length = data.shape[1] - 9
    for i in range(total_length):
        x_tmp = data[:,i:i+9]
        y_tmp = data[9,i+9]
        if valid2(x_tmp, y_tmp, pm25mean, pm25std, a):
            x.append(x_tmp.reshape(-1,))
            y.append(y_tmp)
    # x 會是一個(n, 18, 9)的陣列， y 則是(n, 1) 
    x = np.array(x)
    y = np.array(y)
    return x,y
def get_mean_std(data1, data2):
    pm25 = np.concatenate((data1[9,:], data2[9,:]), axis=None)
    pm25 = [i for i in pm25 if i > 2 and i <= 100]
    pm25_mean = np.mean(pm25)
    pm25_std = np.std(pm25)
    return(pm25_mean,pm25_std)

def add_constant(x):
    return(np.c_[x, np.ones(x.shape[0])])

def train(x, y, l, i):
    xss = x
    yss = y
    num_data, num_feature = xss.shape
    ws = np.zeros(num_feature)
    b = np.zeros(num_feature)
    lr = l
    it = i
    prev_gra = np.zeros(num_feature)
    for j in range(10):
        for i in range(it):
            predict = np.dot(xss,ws)
            diffs = yss - predict
            grad = np.dot(xss.transpose(),diffs) * (2)
            prev_gra += grad**2
            ada = np.sqrt(prev_gra)

            ws = np.add(ws, lr * grad/ada)
            rmse = np.sqrt(np.sum(diffs**2)/num_data)
        # print('iteration group:',j)
        # print('RMSE:',rmse)
    return(ws, rmse)

def expand_dataset(trainxs, trainys):
    train_x = np.vstack(trainxs)
    train_y = np.hstack(trainys)
    return train_x, train_y

if __name__ == "__main__":
	test_name, out_name = sys.argv[1], sys.argv[2]
    
	year1_pd = pd.read_csv('./data/year1-data.csv')
	year2_pd = pd.read_csv('./data/year2-data.csv')

	year1 = readdata(year1_pd)
	year2 = readdata(year2_pd)

	train_data1 = extract(year1)
	train_data2 = extract(year2)

	pm25mean, pm25std = get_mean_std(train_data1, train_data2)

	train_x1, train_y1 = parse2train2(train_data1, pm25mean, pm25std, 3.6)
	train_x2, train_y2 = parse2train2(train_data2, pm25mean, pm25std, 3.6)

	train_x, train_y = expand_dataset((train_x1, train_x2), (train_y1, train_y2))
	train_x = add_constant(train_x)

	w, rmse = train(train_x, train_y, 0.01, 10000)
	np.save('./data/weight.npy', w)