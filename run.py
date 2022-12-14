from pathlib import Path
import numpy as np
import pandas as pd


def read_data(data_dir, CITY, TOTAL, year):
    paths = [path for path in Path(data_dir).glob('*.csv') if year in path.name]
    N = len(paths)

    I_list, R_list = [], []
    for i in range(N):
        path = paths[i]
        data = np.array(pd.read_csv(str(path)))
        h, w = data.shape
        I_val, R_val = 0, 0
        for j in range(h):
            city = data[j, 2]
            if city == CITY:
                I_val = I_val + data[j, 7]
                R_val = R_val + data[j, 9]
        I_list.append(I_val)
        R_list.append(R_val)

    E_list, S_list = [], []
    for i in range(N - 7):
        E_val = I_list[i + 7]
        S_val = TOTAL - I_list[i] - R_list[i] - E_val
        E_list.append(E_val)
        S_list.append(S_val)
    I_list = I_list[:-7]
    R_list = R_list[:-7]

    N = len(I_list)
    dI_list, dR_list, dE_list, dS_list = [], [], [], []
    for i in range(N - 1):
        dI_list.append(I_list[i + 1] - I_list[i])
        dR_list.append(R_list[i + 1] - R_list[i])
        dE_list.append(E_list[i + 1] - E_list[i])
        dS_list.append(S_list[i + 1] - S_list[i])
    I_list.pop()
    R_list.pop()
    E_list.pop()
    S_list.pop()
    assert len(I_list) == len(R_list) == len(E_list) == len(S_list) == len(dI_list) == len(dR_list) == len(dE_list) == len(dS_list)
    return I_list, R_list, E_list, S_list, dI_list, dR_list, dE_list, dS_list


TOTAL = 8804190
CITY = 'New York'
data_dir = 'C:/Users/sunxin/Desktop/COVID-19-master/csse_covid_19_data/csse_covid_19_daily_reports'

I_list, R_list, E_list, S_list, dI_list, dR_list, dE_list, dS_list = read_data(data_dir, CITY, TOTAL, year='2020')
I_list_t, R_list_t, E_list_t, S_list_t, dI_list_t, dR_list_t, dE_list_t, dS_list_t = read_data(data_dir, CITY, TOTAL, year='2021')
print(len(I_list), len(I_list_t))
