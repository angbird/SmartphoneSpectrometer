import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
import re

# 1. 读取数据
df = pd.read_csv('2025_1_12_d8_tp_data_OUTPUT.csv', header=None)

# 2. 第一列是标签，后面是光谱
labels = df.iloc[:, 0]
X = df.iloc[:, 1:].values

# 3. 提取y（浓度）
def extract_conc(label):
    match = re.search(r'tp([0-9.]+)', str(label))
    return float(match.group(1)) if match else np.nan

y = np.array([extract_conc(label) for label in labels])

# 4. 过滤无法解析的样本
mask = ~np.isnan(y)
X = X[mask]
y = y[mask]
labels = labels[mask].reset_index(drop=True)

# 5. PLSR建模
pls = PLSRegression(n_components=2)
y_pred = pls.fit(X, y).predict(X).ravel()
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)

print(f'R2: {r2:.4f}')
print(f'RMSE: {rmse:.4f}')

# 6. 可视化
plt.scatter(y, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('True concentration')
plt.ylabel('Predicted concentration')
plt.title(f'PLSR Predicted vs True (R2={r2:.3f}, RMSE={rmse:.3f})')
plt.show()

# 7. 导出可视化点的数据
df_plot = pd.DataFrame({
    'Sample_label': labels,
    'True_concentration': y,
    'Predicted_concentration': y_pred
})
df_plot.to_csv('plsr_pred_vs_true.csv', index=False)
print('点数据已导出为 plsr_pred_vs_true.csv')



# 计算每个真实浓度下，预测值的均值和标准差
df_plot = pd.DataFrame({'True': y, 'Pred': y_pred})
grouped = df_plot.groupby('True').agg(['mean', 'std'])
x = grouped.index.values
y_mean = grouped['Pred']['mean'].values
y_std = grouped['Pred']['std'].values

plt.errorbar(x, y_mean, yerr=y_std, fmt='o', ecolor='red', capsize=5, label='Mean ± STD')
plt.plot([x.min(), x.max()], [x.min(), x.max()], 'r--', label='y=x')
plt.xlabel('True concentration')
plt.ylabel('Predicted concentration')
plt.title('PLSR Predicted vs True (with error bars)')
plt.legend()
plt.show()

# 如果要保存
plt.savefig('plsr_pred_vs_true_with_errorbar.png', dpi=300)
print('图像已保存为 plsr_pred_vs_true_with_errorbar.png')