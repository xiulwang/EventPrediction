import pymysql
import numpy as np
import pandas as pd
from parameters import *
import scipy.io as sio
from scipy.stats import kurtosis,skew
from pyts.transformation import PAA
import pywt
import os
import multiprocessing as mul


table_name = 'us_feature'
seq_len = 60
fft_len = 14
time_interval = 6
column_list = ['geo0_quad4_count_num','geo0_quad4_gold_avg']


def get_coun_list():
    """get state list from us_feature table"""
    conn = pymysql.connect(host=host,user=user,passwd=passwd,db=db_gdelt,port=port)
    sql = "select distinct coun from "+table_name
    coun_list = pd.read_sql_query(sql,conn)
    conn.close()
    return coun_list.coun


def csv_generate():
    """get original time series data of 50 states"""
    coun_list = get_coun_list()
    conn = pymysql.connect(host=host, user=user, passwd=passwd, db=db_gdelt, port=port)
    for coun in coun_list:
        if coun not in ['US', 'US00', 'US08', 'US49', 'US26', 'USMH', 'USDC', 'USPR']:
            print(coun)
            sql = "select * from "+table_name+" WHERE coun = \'%s\' " % coun
            df = pd.read_sql_query(sql,conn)
            df.to_csv('data/csv_data/'+coun+".csv")


def autocorrelation(x,lags=7):
    n = len(x)
    x = x.reshape(x.shape[0],)
    result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]/(x[i:].std()*x[:n-i].std()*(n-i)) for i in range(1,lags+1)]
    return result


def get_feature(series_data):
    """statistical features"""
    mean = np.mean(series_data)
    variance = np.var(series_data)
    maxvalue = np.max(series_data)
    minvalue = np.min(series_data)
    skewvalue = skew(series_data)[0]
    kurt = kurtosis(series_data)[0]

    """time series features"""
    correlation = autocorrelation(series_data)
    paa = PAA(window_size=None, output_size=21, overlapping=True)
    X_paa = paa.transform(series_data.transpose())
    X_paa = X_paa.reshape(X_paa.shape[1],).tolist()

    """frequency related features"""
    FFT = np.fft.fft(series_data,n=fft_len,axis=0)
    fs = FFT.real.reshape(FFT.shape[0],).tolist()
    fs.extend(FFT.imag.reshape(FFT.shape[0],).tolist())
    ca,cd = pywt.dwt(series_data.transpose(),'db3','smooth')

    """aggregate all features"""
    feature = [mean,variance,maxvalue,minvalue,skewvalue,kurt]
    feature.extend(correlation)
    feature.extend(X_paa)
    feature.extend(fs)
    feature.extend(ca.reshape(ca.shape[1],).tolist())
    return feature


def genereate(df,seq_len,time_interval):
    """
    :param df: original time series data of each state
    :param seq_len: length of the historical data
    :param time_interval:6 denotes the next 1st day label,2nd label,3rd label,...,7th label
    :return: X,Y,dates, Y.shape: [T,time_interval+1]
    """
    batch_size = df.shape[0]
    # scaler = MinMaxScaler()
    # df[column_list] = scaler.fit_transform(df[column_list])
    """log1p function(or divided by 10), others original"""
    df[['geo0_quad4_count_num']] /= 10
    X = [[0]]*(batch_size-seq_len-time_interval)
    Y = [[0]]*(batch_size-seq_len-time_interval)
    Dates = []
    for i in range(seq_len,batch_size-time_interval):
        flag = 1
        for col in column_list:
            series_data = np.array(df[col][i - seq_len:i]).reshape(seq_len, 1)
            series_data = get_feature(series_data)
            if flag:
                X[i - seq_len] = series_data
                Y[i - seq_len] = [df[col][i]]
                flag = 0
            else:
                X[i - seq_len].extend(series_data)
        for interval in range(1, time_interval + 1):
            Y[i - seq_len].extend([df[column_list[0]][i + interval]])
        Dates.append(str(df['dt'][i]))
    return X,Y,Dates


if __name__ == "__main__":
    csv_generate()
    csv_list = os.listdir('data/csv_data/')
    for csv_file in csv_list:
        print(csv_file)
        data = pd.read_csv('data/csv_data/' + csv_file)
        data.drop(['Unnamed: 0', 'index'], axis=1, inplace=True)
        data.sort_values(by='dt', inplace=True)
        data.index = range(0, len(data))
        X, Y, Dates = genereate(data, seq_len, time_interval)
        sio.savemat('data/test-catetory4/' + csv_file[:4], {
            'Data': X,
            'Y': Y,
            'Dates': Dates
        })