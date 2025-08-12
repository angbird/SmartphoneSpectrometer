import matplotlib.pyplot as plt
import csv


# 获取红光数据
with open('Intensity_red_638nm.txt', 'r') as file:
    # 将文件的每一行保存在列表中
    lines = file.readlines()
    for line in lines:
        data_str = line
    # 除去字符串首尾的[ ] 号
    data_str = data_str.replace("[", "").replace("]", "")
    # 获得光谱强度数据
    red_intensity_data = [float(item.strip()) for item in data_str.split(",")]

# 获取绿光数据
with open('Intensity_green_515nm.txt', 'r') as file:
    # 将文件的每一行保存在列表中
    lines = file.readlines()
    for line in lines:
        data_str = line
    # 除去字符串首尾的[ ] 号
    data_str = data_str.replace("[", "").replace("]", "")
    # 获得光谱强度数据
    green_intensity_data = [float(item.strip()) for item in data_str.split(",")]

# 获取蓝光数据
with open('Intensity_blue_450nm.txt', 'r') as file:
    # 将文件的每一行保存在列表中
    lines = file.readlines()
    for line in lines:
        data_str = line
    # 除去字符串首尾的[ ] 号
    data_str = data_str.replace("[", "").replace("]", "")
    # 获得光谱强度数据
    blue_intensity_data = [float(item.strip()) for item in data_str.split(",")]


# 像素值
    pixels = [i + 1 for i in range(1080)]
    # 像素到波长转换
    wavelengths = []
    for i in range(len(pixels)):
        temp = 0.792834*pixels[i] + 124.438196
        # temp = -0.000840 * pixels[i] * pixels[i] + 1.698925 * pixels[i] - 111.374909
        wavelengths.append(temp)

    # 创建一张图
    plt.figure()
    plt.plot(pixels, red_intensity_data, color='red', label='red_intensity-638nm')
    plt.plot(pixels, green_intensity_data, color='green', label='green_intensity-515nm')
    plt.plot(pixels, blue_intensity_data, color='blue', label='blue_intensity-450nm')
    plt.xlabel("pixels")
    # plt.xlim(410, 672)
    plt.ylabel("Intensity")
    plt.axvline(416, color='black', linestyle='--')
    plt.axvline(485, color='black', linestyle='--')
    plt.axvline(650, color='black', linestyle='--')
    plt.annotate('416pixel', xy=(416, 60))
    plt.annotate('485pixel', xy=(485, 40))
    plt.annotate('650pixel', xy=(650, 30))
    # plt.title("wavelength range {:.2f}nm~{:.2f}nm".format(wavelengths[380 - 1], wavelengths[710 - 1]))
    plt.legend()
    plt.show()





    #
    # # 保存数据到CSV文件
    # csv_filename = "laser_calibration.csv"
    # with open(csv_filename, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["Pixel", "Wavelength", "Red Intensity", "Green Intensity", "Blue Intensity"])
    #     for i in range(len(pixels)):
    #         writer.writerow(
    #             [pixels[i], wavelengths[i], red_intensity_data[i], green_intensity_data[i], blue_intensity_data[i]])
    #
    # print(f"数据已保存至 {csv_filename}")

