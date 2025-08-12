
import matplotlib.pyplot as plt
import pandas as pd  # 数据处理库
import pywt
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from scipy.signal import savgol_filter
import joblib
from sklearn.ensemble import RandomForestRegressor
from kymatio import Scattering1D

# SPA连续投影算法
def successive_projections_algorithm(X, num_variables):
    num_samples, num_features = X.shape
    selected_variables = []
    P = np.eye(num_samples)
    for i in range(num_variables):
        var_projections = np.dot(X.T, P).T
        var_norms = np.sum(var_projections ** 2, axis=0)
        next_var = np.argmax(var_norms)
        selected_variables.append(next_var)
        xi = X[:, [next_var]]
        P = P - np.dot(np.dot(P, xi), np.dot(xi.T, P)) / np.dot(np.dot(xi.T, P), xi)
    return selected_variables


# 进行多元散射校正（MSC）
def msc(input_data):
    # 计算平均光谱作为参考
    ref_spectrum = np.mean(input_data, axis=0)
    corrected_data = np.zeros_like(input_data)

    for i in range(input_data.shape[0]):
        # 对每个样本进行最小二乘线性回归
        fit = np.polyfit(ref_spectrum, input_data[i, :], 1)
        # 应用校正
        corrected_data[i, :] = (input_data[i, :] - fit[1]) / fit[0]

    return corrected_data, ref_spectrum, fit

# 使用SG进行滤波
def sg_filter(X, m, p, d):
    X_filtered = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_filtered[i, :] = savgol_filter(X[i, :], window_length=m, polyorder=p, deriv=d)
    return X_filtered

# 定义 SNV 预处理函数，多样本
def snv(X):
    X_mean = np.mean(X, axis=1, keepdims=True)  # 计算每个样本的均值
    X_std = np.std(X, axis=1, keepdims=True)    # 计算每个样本的标准差
    X_snv = (X - X_mean) / X_std                 # 对每个样本进行 SNV 处理
    return X_snv


# def min_max_normalize(data):
#     min_val = np.min(data)
#     max_val = np.max(data)
#     normalized_data = (data - min_val) / (max_val - min_val)
#     return normalized_data

def CT(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after MeanScaler :(n_samples, n_features)
       """
    for i in range(data.shape[0]):
        MEAN = np.mean(data[i])
        data[i] = data[i] - MEAN
    return data

# ks算法划分数据集
def ks(x, y, test_size=0.3):
    """
    通过最大化训练样本之间的最小距离来选择具有代表性的样本作为训练集

    :param x: shape (n_samples, n_features)
    :param y: shape (n_sample, )
    :param test_size: the ratio of test_size (float)
    :return: spec_train: (n_samples, n_features)
             spec_test: (n_samples, n_features)
             target_train: (n_sample, )
             target_test: (n_sample, )
    """
    M = x.shape[0]
    N = round((1 - test_size) * M)
    samples = np.arange(M)

    D = np.zeros((M, M))

    for i in range((M - 1)):
        xa = x[i, :]
        for j in range((i + 1), M):
            xb = x[j, :]
            D[i, j] = np.linalg.norm(xa - xb)

    maxD = np.max(D, axis=0)
    index_row = np.argmax(D, axis=0)
    index_column = np.argmax(maxD)

    m = np.zeros(N)
    m[0] = np.array(index_row[index_column])
    m[1] = np.array(index_column)
    m = m.astype(int)
    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]

    for i in range(2, N):
        pool = np.delete(samples, m[:i])
        dmin = np.zeros((M - i))
        for j in range((M - i)):
            indexa = pool[j]
            d = np.zeros(i)
            for k in range(i):
                indexb = m[k]
                if indexa < indexb:
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)
        dminmax[i] = np.max(dmin)
        index = np.argmax(dmin)
        m[i] = pool[index]

    m_complement = np.delete(np.arange(x.shape[0]), m)

    spec_train = x[m, :]
    target_train = y[m]
    spec_test = x[m_complement, :]
    target_test = y[m_complement]
    return spec_train, spec_test, target_train, target_test


def spxy(data, label, test_size=0.3):
    """
    SPXY算法划分数据集

    SPXY算法是一种基于样本相似度和标签相似度的数据集划分方法。它通过最小化训练集样本之间的距离和标签差异来选择具有代表性的样本作为训练集
    SPXY算法的主要步骤如下:
    计算样本之间的欧氏距离和标签差异,构建距离矩阵D和标签差异矩阵Dy。
    对距离矩阵进行归一化处理。
    选择距离最远的两个样本作为初始训练样本。
    迭代选择距离已选训练样本最远的样本作为新的训练样本,直到达到指定的训练样本数量。
    将未被选为训练样本的样本作为测试样本。
    原文链接：https://blog.csdn.net/qq_42629547/article/details/138381450

    参数:
    data: 特征数据,shape为(n_samples, n_features)
    label: 标签数据,shape为(n_samples,)
    test_size: 测试集比例,默认为0.3

    返回:
    X_train: 训练集特征,shape为(n_samples, n_features)
    X_test: 测试集特征,shape为(n_samples, n_features)
    y_train: 训练集标签,shape为(n_samples,)
    y_test: 测试集标签,shape为(n_samples,)
    """
    # 备份数据
    x_backup = data
    y_backup = label

    # 样本数量
    M = data.shape[0]
    # 训练样本数量
    N = round((1 - test_size) * M)
    # 样本索引
    samples = np.arange(M)

    # 标准化标签
    label = (label - np.mean(label)) / np.std(label)

    # 初始化距离矩阵
    D = np.zeros((M, M))
    Dy = np.zeros((M, M))

    # 计算样本之间的欧氏距离和标签差异
    for i in range(M - 1):
        xa = data[i, :]
        ya = label[i]
        for j in range((i + 1), M):
            xb = data[j, :]
            yb = label[j]
            D[i, j] = np.linalg.norm(xa - xb)
            Dy[i, j] = np.linalg.norm(ya - yb)

    # 归一化距离矩阵
    Dmax = np.max(D)
    Dymax = np.max(Dy)
    D = D / Dmax + Dy / Dymax

    # 选择距离最远的两个样本作为初始训练样本
    maxD = D.max(axis=0)
    index_row = D.argmax(axis=0)
    index_column = maxD.argmax()

    m = np.zeros(N)
    m[0] = index_row[index_column]
    m[1] = index_column
    m = m.astype(int)

    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]

    # 迭代选择训练样本
    for i in range(2, N):
        pool = np.delete(samples, m[:i])
        dmin = np.zeros(M - i)
        for j in range(M - i):
            indexa = pool[j]
            d = np.zeros(i)
            for k in range(i):
                indexb = m[k]
                if indexa < indexb:
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)
        dminmax[i] = np.max(dmin)
        index = np.argmax(dmin)
        m[i] = pool[index]

    # 测试集索引
    m_complement = np.delete(np.arange(data.shape[0]), m)

    # 划分训练集和测试集
    X_train = data[m, :]
    y_train = y_backup[m]
    X_test = data[m_complement, :]
    y_test = y_backup[m_complement]

    return X_train, X_test, y_train, y_test

def scattering_transform(X, J=3, Q=8):
    scattering = Scattering1D(J=J, shape=(X.shape[1],), Q=Q)
    X_scattered = []

    for x in X:
        Sx = scattering(x)
        # Sx = Sx[:1]
        # Sx = Sx[:23]
        # Sx = np.delete(Sx, order0[0], axis=0)  #移除零阶散射系数
        # print(Sx.shape)
        X_scattered.append(Sx.flatten())

    # 将所有展平后的结果转换为 numpy 数组，形状为 (n_samples, n_features)
    X_scattered = np.array(X_scattered)
    return X_scattered

# DWT小波变换去噪
def wavelet_denoising(data, wavelet, level):
    coeff = pywt.wavedec(data, wavelet, mode='per', level=level)
    sigma = np.median(np.abs(coeff[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')

if __name__ == '__main__':
    print("hello world")








