from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 超参数
INTERVAL = 1 # 采样间隔
N = 8804190 # 总人数
CITY = 'New York' # 城市
data_dir = './COVID-19-master/csse_covid_19_data/csse_covid_19_daily_reports'
fit_rate = 0.7 # 用于拟合的数据量所占的比例
sample_year = '2020' # 选用的年份
sample_months = ['01', '02', '03', '04', '05', '06', '07'] # 选用的月份

# 读路径
paths = [path for path in Path(data_dir).glob('*.csv') if sample_year in path.name]
paths.sort()
length = len(paths)

# 读I R
I, R = [], []
for i in range(length):
    path = paths[i]
    month = path.name.split('-')[0]
    if month not in sample_months:
        continue
    else:
        pass
        # print('reading', path.name)
    data = np.array(pd.read_csv(str(path)))
    h, w = data.shape
    I_val, R_val = 0, 0
    for j in range(h):
        city = data[j, 2]
        if city == CITY:
            I_val = I_val + data[j, 7]
            R_val = R_val + data[j, 9]
    I.append(I_val)
    R.append(R_val)

# 根据I R获得E S
length = len(I)
E, S = [], []
for i in range(length - 7):
    E_val = I[i + 7]
    S_val = N - I[i] - R[i] - E_val
    E.append(E_val)
    S.append(S_val)
I = I[:-7]
R = R[:-7]

# 算出四个变量各自的微分
N = len(I)
dI, dR, dE, dS = [], [], [], []
for i in range(N - 1):
    dI.append((I[i + 1] - I[i]) / INTERVAL)
    dR.append((R[i + 1] - R[i]) / INTERVAL)
    dE.append((E[i + 1] - E[i]) / INTERVAL)
    dS.append((S[i + 1] - S[i]) / INTERVAL)
I.pop()
R.pop()
E.pop()
S.pop()

# 确保所有变量长度一致
assert len(I) == len(R) == len(E) == len(S) == len(dI) == len(dR) == len(dE) == len(dS)

def print_np_info(name, m):
    print(name, m.shape, 'min:{}, max:{}, mean:{}'.format(np.min(m), np.max(m), np.mean(m)))

length = len(I)
fit_num = int(length * fit_rate) # 用于拟合的数目

# 用于拟合的数据
S1 = S[:fit_num]
E1 = E[:fit_num]
I1 = I[:fit_num]
R1 = R[:fit_num]
dS1 = dS[:fit_num]
dE1 = dE[:fit_num]
dI1 = dI[:fit_num]
dR1 = dR[:fit_num]

# 用于测试的数据
S2 = S[fit_num:]
E2 = E[fit_num:]
I2 = I[fit_num:]
R2 = R[fit_num:]
dS2 = dS[fit_num:]
dE2 = dE[fit_num:]
dI2 = dI[fit_num:]
dR2 = dR[fit_num:]

# 把拟合数据整理为最小二乘法方程的矩阵形式 XA = Y. 其中，A为所求参数，Y为微分
T1 = np.array([i for i in range(0, fit_num, INTERVAL)])
Z1 = np.zeros((fit_num, 1))
O1 = np.ones((fit_num, 1))
S1 = np.expand_dims(np.array(S1), axis=1)
E1 = np.expand_dims(np.array(E1), axis=1)
I1 = np.expand_dims(np.array(I1), axis=1)
R1 = np.expand_dims(np.array(R1), axis=1)
dS1 = np.expand_dims(np.array(dS1), axis=1)
dE1 = np.expand_dims(np.array(dE1), axis=1)
dI1 = np.expand_dims(np.array(dI1), axis=1)
dR1 = np.expand_dims(np.array(dR1), axis=1)
tmpA = np.c_[Z1, -I1*S1/N, Z1, Z1]
tmpB = np.c_[-E1, I1*S1/N, Z1, O1]
tmpC = np.c_[E1, Z1, -I1, Z1]
tmpD = np.c_[Z1, Z1, I1, Z1]
X1 = np.r_[tmpA, tmpB, tmpC, tmpD]
Y1 = np.r_[dS1, dE1, dI1, dR1]

# XA = Y -> X'XA = X'Y
# A = inv(X'X)X'Y
A1 = np.linalg.inv(X1.T.dot(X1)).dot(X1.T).dot(Y1) # 计算出参数
a, b, r, w = A1[0], A1[1], A1[2], A1[3]

# 把测试数据整理为最小二乘法方程的矩阵形式
length2 = length - fit_num
T2 = np.array([i for i in range(0, length2, INTERVAL)])
S2 = np.expand_dims(np.array(S2), axis=1)
E2 = np.expand_dims(np.array(E2), axis=1)
I2 = np.expand_dims(np.array(I2), axis=1)
R2 = np.expand_dims(np.array(R2), axis=1)
dS2 = np.expand_dims(np.array(dS2), axis=1)
dE2 = np.expand_dims(np.array(dE2), axis=1)
dI2 = np.expand_dims(np.array(dI2), axis=1)
dR2 = np.expand_dims(np.array(dR2), axis=1)

# 根据模型公式，反算出估计出的S I E
I2_pred = dR2 / r
E2_pred = (dI2 + dR2) / a
S2_pred = (-N * dS2) / (b * I2)

# 真实的S I E与估计的S I E做评价
print('\n\n')
print('a:{}, b:{}, r:{}, w:{}'.format(a, b, r, w))
print("MSE of S:", mean_squared_error(S2, S2_pred))
print("MSE of I:", mean_squared_error(I2, I2_pred))
print("MSE of E:", mean_squared_error(E2, E2_pred))

plt.plot(T2, S2, label="real S")
plt.plot(T2, S2_pred, label="pred S")
plt.legend()
# plt.show()