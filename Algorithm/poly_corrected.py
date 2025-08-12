import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 加载数据
standard_df = pd.read_csv('./data/my_average_data.csv', header=None)
target_df = pd.read_csv('./data/my_average_data_2.csv', header=None)

# 2. 提取像素范围 300-600（注意切片右闭）
standard_data = standard_df.values[:, ]
target_data = target_df.values[:, ]

# 3. 同步划分（确保样本一一对应）
standard_cal, standard_test, target_cal, target_test = train_test_split(
    standard_data, target_data, test_size=0.2, random_state=42
)

# 4. 逐像素多项式拟合
degree = 3  # 二次多项式
poly = PolynomialFeatures(degree=degree, include_bias=False)
coefs = []  # 每个像素的多项式系数 [a, b, intercept]

for i in range(standard_cal.shape[1]):
    x = target_cal[:, i].reshape(-1, 1)
    y = standard_cal[:, i]

    x_poly = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)

    # 将每个像素的 [a, b, intercept] 存入列表
    coefs.append(model.coef_.tolist() + [model.intercept_])

coefs = np.array(coefs)  # shape: (pixels, degree + 1)

# 5. 校正测试集
corrected_test = []

for i in range(target_test.shape[0]):  # 遍历每个样本
    row_corrected = []
    for j in range(target_test.shape[1]):  # 遍历每个像素
        x_val = target_test[i, j]
        coeffs = coefs[j]  # 当前像素的多项式系数 [a, b, intercept]
        y_val = sum([coeffs[k] * x_val**(k+1) for k in range(len(coeffs)-1)]) + coeffs[-1]
        row_corrected.append(y_val)
    corrected_test.append(row_corrected)

corrected_test = np.array(corrected_test)

# 6. 保存校正模型参数
np.savetxt('poly_coefs.csv', coefs, delimiter=',')

# 7. 评估指标
mse_before = mean_squared_error(standard_test, target_test)
mse_after = mean_squared_error(standard_test, corrected_test)

mae_before = mean_absolute_error(standard_test, target_test)
mae_after = mean_absolute_error(standard_test, corrected_test)

print(f"MSE（校正前）: {mse_before:.4f}")
print(f"MSE（校正后）: {mse_after:.4f}")
print(f"MAE（校正前）: {mae_before:.4f}")
print(f"MAE（校正后）: {mae_after:.4f}")

# Pearson 相关系数（按样本计算）
r_before = [pearsonr(standard_test[i, :], target_test[i, :])[0] for i in range(standard_test.shape[0])]
r_after = [pearsonr(standard_test[i, :], corrected_test[i, :])[0] for i in range(standard_test.shape[0])]

print(f"平均 Pearson 相关系数（校正前）: {np.mean(r_before):.4f}")
print(f"平均 Pearson 相关系数（校正后）: {np.mean(r_after):.4f}")

# 8. 可视化（取随机10个样本）
wavelengths = np.arange(0, 1080)
num_plot = 10
indices = np.random.choice(len(standard_test), num_plot, replace=False)

for i, idx in enumerate(indices):
    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, standard_test[idx], label='Standard', color='black')
    plt.plot(wavelengths, target_test[idx], label='Target (raw)', linestyle='--', color='red')
    plt.plot(wavelengths, corrected_test[idx], label='Target (corrected)', linestyle=':', color='blue')
    plt.title(f'Sample #{idx}')
    plt.xlabel('Pixel')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
