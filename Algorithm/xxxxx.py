import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def read_data(standard_path, target_path):
    """读取标准数据和目标数据"""
    try:
        standard_df = pd.read_csv(standard_path, header=None)
        target_df = pd.read_csv(target_path, header=None)
        standard_data = standard_df.values[:, 300:700]
        target_data = target_df.values[:, 300:700]
        return standard_data, target_data
    except Exception as e:
        print(f"读取数据时出错: {e}")
        return None, None


def calculate_sorted_calibration_coefficients(standard_data, target_data,
                                              pixel_range=(300, 699),
                                              intensity_bins=201):
    """先排序每列数据再计算校正系数"""
    min_pixel, max_pixel = pixel_range
    n_pixels = max_pixel - min_pixel + 1
    intensity_range = (0, 200)
    min_intensity, max_intensity = intensity_range
    coef_matrix = np.zeros((n_pixels, intensity_bins))
    count_matrix = np.zeros((n_pixels, intensity_bins))

    for p in range(n_pixels):
        # 对当前像素的标准值和目标值排序
        std_values = standard_data[:, p]
        tgt_values = target_data[:, p]
        sort_idx = np.argsort(tgt_values)
        sorted_std = std_values[sort_idx]
        sorted_tgt = tgt_values[sort_idx]

        # 计算排序后的校正系数
        valid_mask = sorted_tgt > 0
        sorted_coef = sorted_std[valid_mask] / sorted_tgt[valid_mask]
        sorted_tgt = sorted_tgt[valid_mask]

        # 将系数分配到强度分箱
        for tgt, coef in zip(sorted_tgt, sorted_coef):
            bin_idx = int(np.round(tgt))
            if min_intensity <= bin_idx <= max_intensity:
                bin_idx -= min_intensity
                coef_matrix[p, bin_idx] += coef
                count_matrix[p, bin_idx] += 1

    # 计算分箱平均系数并平滑处理
    mask = count_matrix > 0
    coef_matrix[mask] /= count_matrix[mask]
    smoothed_coef = gaussian_filter(coef_matrix, sigma=1.0)
    coef_matrix[~mask] = smoothed_coef[~mask]

    return coef_matrix


def create_heatmap(coefs, pixel_range=(200, 700), intensity_range=(0, 200),
                   cmap='viridis', save_path=None):
    """创建校正系数热力图"""
    min_pixel, max_pixel = pixel_range
    min_intensity, max_intensity = intensity_range

    plt.figure(figsize=(12, 10))
    im = plt.imshow(coefs, cmap=cmap, aspect='auto',
                    extent=[min_pixel, max_pixel, min_intensity, max_intensity],
                    origin='lower', vmin=0.8, vmax=1.2)
    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label('校正系数', fontsize=12, rotation=270, labelpad=20)

    plt.xlabel('像素位置', fontsize=12)
    plt.ylabel('强度值', fontsize=12)
    plt.title('校正系数热力图 (像素200-700, 强度0-200)', fontsize=14, pad=10)
    plt.xticks(np.arange(min_pixel, max_pixel + 1, 20))
    plt.yticks(np.arange(min_intensity, max_intensity + 1, 20))
    plt.grid(which='major', color='w', linestyle='-', linewidth=0.5, alpha=0.3)

    # 统计信息计算
    valid_coefs = coefs[coefs > 0]
    avg_coef = np.mean(valid_coefs)
    max_coef = np.max(valid_coefs)
    min_coef = np.min(valid_coefs)
    max_pos = np.unravel_index(np.argmax(coefs), coefs.shape)
    min_flat_idx = np.flatnonzero(coefs > 0)[np.argmin(valid_coefs)]
    min_pixel_idx, min_intensity_idx = np.unravel_index(min_flat_idx, coefs.shape)

    stats_text = f"平均校正系数: {avg_coef:.4f}\n"
    stats_text += f"最大校正系数: {max_coef:.4f} (像素{min_pixel + max_pos[0]}, 强度{min_intensity + max_pos[1]})\n"
    stats_text += f"最小校正系数: {min_coef:.4f} (像素{min_pixel + min_pixel_idx}, 强度{min_intensity + min_intensity_idx})"

    plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    else:
        plt.show()
    return plt.gcf()


def main():
    """主函数"""
    standard_path = './PhonesData/消除后处理_调整光源强度/xiaomi14/allData.csv'
    target_path = './PhonesData/消除后处理_调整光源强度/redmik70/allData.csv'

    standard_data, target_data = read_data(standard_path, target_path)
    if standard_data is None or target_data is None:
        print("数据读取失败，程序退出")
        return

    print("正在计算排序后的校正系数...")
    pixel_range = (300, 699)
    coef_matrix = calculate_sorted_calibration_coefficients(
        standard_data, target_data, pixel_range=pixel_range
    )

    print("正在生成热力图...")
    display_pixel_range = (200, 700)
    offset = 300 - 200
    display_coef = np.zeros((display_pixel_range[1] - display_pixel_range[0] + 1, 201))
    valid_pixels = min(400, coef_matrix.shape[0])  # 显示400个像素范围
    display_coef[offset:offset + valid_pixels, :] = coef_matrix[:valid_pixels, :]

    create_heatmap(display_coef, display_pixel_range)


if __name__ == "__main__":
    main()