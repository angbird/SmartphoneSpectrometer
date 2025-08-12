import numpy as np
import matplotlib.pyplot as plt

# 示例数据
pixels = np.array([416, 485, 650])
wavelength = np.array([450, 515, 638])

# 一次多项式拟合
coefficients = np.polyfit(pixels, wavelength, 1)

# 生成拟合后的曲线
# x_fit = np.linspace(min(pixels), max(pixels), 100)
x_fit = np.linspace(380, 710, 330)
y_fit = np.polyval(coefficients, x_fit)

# 输出拟合公式
k, b = coefficients  # 提取系数
# a, b, c = coefficients  # 二次拟合系数
fit_equation = "y={:.6f}x + {:.6f}".format(k, b)  # 构建拟合公式的字符串表示
# fit_equation = "y={:.6f}x^2 + {:.6f}x + {:.6f}".format(a, b, c)  # 构建拟合公式的字符串表示
print("拟合公式:", fit_equation)

# 计算总平方和
total_sum_squares = np.sum((wavelength - np.mean(wavelength))**2)
# 计算残差平方和（Residual Sum of Squares）
residual_sum_squares = np.sum((wavelength - np.polyval(coefficients, pixels))**2)
# 计算决定系数
r_squared = 1 - (residual_sum_squares / total_sum_squares)


# 绘制原始数据和拟合曲线
plt.scatter(pixels, wavelength, label='raw data')
plt.plot(x_fit, y_fit, label='fitted curves', color='red')
plt.legend()
plt.xlabel('pixels')
plt.ylabel('wavelength')
plt.annotate(fit_equation, xy=(500, 500))
plt.annotate("R²:{:.6f}".format(r_squared), xy=(500, 450))
plt.show()

print(y_fit[0])
print(y_fit[329])
