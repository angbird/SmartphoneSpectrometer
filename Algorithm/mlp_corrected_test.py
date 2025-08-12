import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# 1. 加载数据
standard_df = pd.read_csv('./data/my_average_data_d8.csv', header=None)
target_df = pd.read_csv('./data/my_average_data_d13.csv', header=None)
standard_test = pd.read_csv("./data/data20250529/lyt/d8-fbp.csv", header=None)
target_test = pd.read_csv("./data/data20250529/lyt/d13-fbp.csv", header=None)

# 2. 提取像素范围
standard_data = standard_df.values[:, ]
target_data = target_df.values[:, ]
standard_data_test = standard_test.values[:, ]
target_data_test = target_test.values[:, ]
wavelengths = np.arange(0, 1080)

# standard_cal, standard_test, target_cal, target_test = train_test_split(standard_data, target_data, test_size=0.2, random_state=42)

standard_cal = standard_data
target_cal = target_data
standard_test = standard_data_test
target_test = target_data_test

print(f"校正集: {len(standard_cal)}, 测试集: {len(standard_test)}")

# 4. MLP 模型构建与训练
mlp = MLPRegressor(hidden_layer_sizes=(128, 64),  # 可调结构
                   activation='relu',
                   solver='adam',
                   max_iter=2000,
                   random_state=42)

mlp.fit(target_cal, standard_cal)

# 5. 校正目标测试集
corrected_test = mlp.predict(target_test)

# 6. 保存校正结果（可选）
# np.savetxt('mlp_corrected_test.csv', corrected_test, delimiter=',')

# 7. 评估指标
mse_before = mean_squared_error(standard_test, target_test)
mse_after = mean_squared_error(standard_test, corrected_test)

mae_before = mean_absolute_error(standard_test, target_test)
mae_after = mean_absolute_error(standard_test, corrected_test)

r_before = [pearsonr(standard_test[i], target_test[i])[0] for i in range(len(standard_test))]
r_after = [pearsonr(standard_test[i], corrected_test[i])[0] for i in range(len(standard_test))]

print(f"MSE（校正前）:  {mse_before:.4f}")
print(f"MSE（校正后）:  {mse_after:.4f}")
print(f"MAE（校正前）:  {mae_before:.4f}")
print(f"MAE（校正后）:  {mae_after:.4f}")
print(f"平均 Pearson（校正前）: {np.mean(r_before):.4f}")
print(f"平均 Pearson（校正后）: {np.mean(r_after):.4f}")

plt.figure(figsize=(12, 6))
# 只给第一条线加图例标签，其他的设 label=None 避免重复
for i in range(standard_test.shape[0]):
    label = 'Standard (Ground Truth)' if i == 0 else None
    plt.plot(wavelengths, standard_test[i], color='black', alpha=0.3, label=label)

for i in range(target_test.shape[0]):
    label = 'Target (Before Correction)' if i == 0 else None
    plt.plot(wavelengths, target_test[i], color='red', alpha=0.3, label=label)

for i in range(corrected_test.shape[0]):
    label = 'Target (After Correction)' if i == 0 else None
    plt.plot(wavelengths, corrected_test[i], color='green', alpha=0.3, label=label)

plt.xlabel('Wavelength Index')
plt.ylabel('Intensity')
plt.title('Spectral Correction Result')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
