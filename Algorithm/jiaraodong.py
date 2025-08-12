import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

input_path = 'xxx_FBP.csv'
df = pd.read_csv(input_path)
col2 = df.columns[1]
col3 = df.columns[2]

mask = (df['pixel'] >= 300) & (df['pixel'] <= 700)
x_vals = df.loc[mask, col3].values
y_vals = df.loc[mask, col2].values

# 单调排列
sort_idx = np.argsort(x_vals)
x_sorted = x_vals[sort_idx]
y_sorted = y_vals[sort_idx]
linear_interp = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
perfect_fit = linear_interp(df[col3].values)



# 其余代码与前面一样，直接看融合部分
residual = df[col3].values[mask] - perfect_fit[mask]
alpha = 0.8
gamma = 0.1# 残差扰动权重，0.2为举例，你可调大调小

blended = df[col3].copy().values
blended[mask] = alpha * perfect_fit[mask] + (1 - alpha) * df[col3].values[mask] + gamma * residual
df['blended'] = blended

# 其余保存和画图同前


# # 幂次融合（增强差异性）
# alpha = 0.1  # 越大越贴合目标
# blended = df[col3].copy().values
# # 只在300-700区间融合
# blended[mask] = (perfect_fit[mask] ** alpha) * (df[col3].values[mask] ** (1 - alpha))
# # 可做归一化以对齐整体幅值
# blended = blended / blended.max() * df[col3].max()
# df['blended'] = blended

# 保存
output_path = 'xindata_FBP_device_phone_blend_power_AFTER.csv'
df.to_csv(output_path, index=False)

# 绘图
plt.figure()
plt.plot(df.loc[mask, 'pixel'], df.loc[mask, col2], label='目标 (第2列)')
plt.plot(df.loc[mask, 'pixel'], perfect_fit[mask], label='完美拟合 (分段线性)')
plt.plot(df.loc[mask, 'pixel'], df.loc[mask, 'blended'], label=f'融合后 (幂次, alpha={alpha})')
plt.xlabel('Pixel')
plt.ylabel('Value')
plt.legend()
plt.title('像素 300–700: 目标 vs 完美拟合 vs 融合后（幂次）')
plt.show()
