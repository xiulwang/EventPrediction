from sklearn.linear_model import LinearRegression,Lasso
import numpy as np
import scipy.io as sio
import os
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

data_fold = 'data/test-catetory4/'


def get_data_split():
    country_list = os.listdir(data_fold)
    index = np.zeros((len(country_list), 1))
    testIndex = np.zeros((len(country_list), 1))
    mat = sio.loadmat(data_fold + str(country_list[0]))
    X = mat['Data']
    X = X.reshape(X.shape[0], X.shape[1])
    Y = mat['Y']
    ind = int(X.shape[0]*0.8)
    index[0] = ind
    testIndex[0] = X.shape[0] - ind
    trainX = X[:ind]
    trainY = Y[:ind]
    testX = X[ind:]
    testY = Y[ind:]
    for i in range(1,len(country_list)):
        mat = sio.loadmat(data_fold + str(country_list[i]))
        X = mat['Data']
        X = X.reshape(X.shape[0], X.shape[1])
        Y = mat['Y']
        ind = int(X.shape[0] * 0.8)
        index[i] = index[i - 1] + ind
        testIndex[i] = testIndex[i - 1] + X.shape[0] - ind
        trainX = np.concatenate([trainX, X[:ind]])
        trainY = np.concatenate([trainY, Y[:ind]])
        testX = np.concatenate([testX, X[ind:]])
        testY = np.concatenate([testY, Y[ind:]])
    return trainX,trainY,testX,testY,index,testIndex


def ae(predict,Y_test):
    return np.linalg.norm((predict - Y_test),ord=1) / Y_test.shape[0]


def mse(predict,Y_test):
    return np.linalg.norm((predict - Y_test),ord=2) / np.sqrt(Y_test.shape[0])


def rmse(predict,Y_test,testIndex):
    error = 0
    for i in range(len(testIndex)):
        if i==0:
            length = testIndex[i]
            pre = predict[:int(testIndex[i,0])]
            y_true = Y_test[:int(testIndex[i,0])]
        else:
            length = testIndex[i,0]-testIndex[i-1,0]
            pre = predict[int(testIndex[i-1,0]):int(testIndex[i,0])]
            y_true = Y_test[int(testIndex[i-1,0]):int(testIndex[i,0])]
        error += mse(pre,y_true)*length
    return error/predict.shape[0]


def one_model():
    """train one model use all data, Y_train[:, 0] denotes the next 1st prediction"""
    X_train, Y_train, X_test, Y_test, index, testIndex = get_data_split()
    Y_train = Y_train[:, 0].reshape(-1, 1)
    Y_test = Y_test[:, 0].reshape(-1, 1)
    linreg = LinearRegression()
    linreg.fit(X_train, Y_train)
    predict = linreg.predict(X_test)
    test_cost = ae(predict, Y_test)
    print(test_cost)


def seperate_model():
    """for each state, train a model using corresponding data"""
    X_train, Y_train, X_test, Y_test, index, testIndex = get_data_split()
    Y_train = Y_train[:, 0].reshape(-1, 1)
    Y_test = Y_test[:, 0].reshape(-1, 1)
    J = 0
    for i in range(0, len(testIndex)):
        if i == 0:
            test_X = X_test[:int(testIndex[i])]
            test_Y = Y_test[:int(testIndex[i])]
            train_X = X_train[:int(index[i])]
            train_Y = Y_train[:int(index[i])]
        else:
            test_X = X_test[int(testIndex[i - 1]):int(testIndex[i])]
            test_Y = Y_test[int(testIndex[i - 1]):int(testIndex[i])]
            train_X = X_train[int(index[i - 1]):int(index[i])]
            train_Y = Y_train[int(index[i - 1]):int(index[i])]
        linreg = LinearRegression()
        linreg.fit(train_X, train_Y)
        # clf = RandomForestRegressor(max_depth=2, random_state=0)
        # clf.fit(train_X,train_Y)
        predict1 = linreg.predict(test_X)
        J += np.linalg.norm((predict1 - test_Y), ord=1)
        print(i, np.linalg.norm((predict1 - test_Y), ord=1)/test_Y.shape[0])
    print(J / Y_test.shape[0])


if __name__ == "__main__":
    one_model()
    seperate_model()





