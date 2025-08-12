package com.example.watermonitor;

import android.util.Log;

public class SignalAligner {
    // 定义固定位置
    private static final int FIXED_X_POSITION = 500;   // x方向的目标位置
    private static final float FIXED_Y_POSITION = 48.0f;  // y方向的目标位置
    private static final int TARGET_POSITION = 500;    // 目标位置500
    private static final String TAG = "SignalAligner";

    /**
     * 计算给定信号在 x 方向上的质心位置
     *
     * @param signal 输入信号数组
     * @return 质心位置
     */
    private int calculateCentroidX(float[] signal) {
        float sum = 0;
        float weightedSum = 0;
        for (int i = 0; i < signal.length; i++) {
            weightedSum += i * signal[i];
            sum += signal[i];
        }
        return (int) Math.round(weightedSum / sum);
    }

    /**
     * 计算给定信号在 y 方向上的质心（平均值）
     *
     * @param signal 输入信号数组
     * @return y 方向质心位置
     */
    private float calculateCentroidY(float[] signal) {
        float sum = 0;
        for (float value : signal) {
            sum += value;
        }
        return sum / signal.length;
    }

    /**
     * 将信号的质心对齐到固定位置（x方向和y方向）
     *
     * @param signal 输入信号数组
     * @return 对齐后的信号数组
     */
    public float[] alignToCentroid(float[] signal) {
        // 计算 x 和 y 方向上的质心位置
        int centroidX = calculateCentroidX(signal);
        float centroidY = calculateCentroidY(signal);

        // 计算 x 和 y 方向上的平移量
        int shiftAmountX = FIXED_X_POSITION - centroidX;
        float shiftAmountY = FIXED_Y_POSITION - centroidY;

        Log.d(TAG, "XCentroid: x方向质心: " + centroidX);
        Log.d(TAG, "YCentroid: y方向质心: " + centroidY);
        Log.d(TAG, "alignToCentroid: x方向偏移大小: " + shiftAmountX);
        Log.d(TAG, "alignToCentroid: y方向偏移大小: " + shiftAmountY);

        // 创建修复后的信号数组
        float[] alignedSignal = new float[signal.length];

        // 平移信号
        for (int i = 0; i < signal.length; i++) {
            // 计算 x 方向的平移
            int shiftedIndexX = (i + shiftAmountX) % signal.length;
            if (shiftedIndexX < 0) {
                shiftedIndexX += signal.length; // 处理负索引的情况
            }

            // 在 y 方向应用平移，将每个值加上 shiftAmountY
//            alignedSignal[shiftedIndexX] = signal[i] + shiftAmountY;
            alignedSignal[shiftedIndexX] = signal[i] ;
        }

        return alignedSignal;
    }

    // 检测信号的局部最小值位置，对信号进行左右平移，将最小值的位置固定在指定位置
    public int findLocalMinimumInRange(float[] signal, int startIndex, int endIndex) {
        // 遍历指定范围内的信号，找到局部最小值
        for (int i = startIndex + 1; i < endIndex - 1; i++) {
            // 检查当前点是否是局部最小值
            if (signal[i] < signal[i - 1] && signal[i] < signal[i + 1]) {
                return i; // 找到局部最小值，返回其索引
            }
        }
        return -1; // 如果没有找到局部最小值，返回-1
    }

    /**
     * 将信号对齐到指定局部最小值位置，并固定 x 和 y 方向上的位置
     *
     * @param signal       输入信号数组
     * @param startIndex   起始索引
     * @param endIndex     结束索引
     * @param targetPosition 目标位置
     * @return 对齐后的信号数组
     */
    public float[] alignToLocalMinimum(float[] signal, int startIndex, int endIndex, int targetPosition) {
        // 查找指定范围内的局部最小值位置
        int minIndex = findLocalMinimumInRange(signal, startIndex, endIndex);

        // 如果没有找到局部最小值，直接返回原始信号
        if (minIndex == -1) {
            return signal;
        }

        // 计算平移量
        int shiftAmount = targetPosition - minIndex;

        // 创建平移后的信号数组
        float[] alignedSignal = new float[signal.length];

        // 对信号进行平移
        for (int i = 0; i < signal.length; i++) {
            int shiftedIndex = (i + shiftAmount) % signal.length;
            if (shiftedIndex < 0) {
                shiftedIndex += signal.length; // 处理负索引的情况
            }
            alignedSignal[shiftedIndex] = signal[i];
        }

        return alignedSignal;
    }

    /**
     * 进行二次对齐，首先质心对齐，然后查找区间 [400, 700] 内的局部最小值并进行二次平移
     *
     * @param signal 输入信号数组
     * @return 二次对齐后的信号数组
     */
    public float[] alignAndRefine(float[] signal) {
        // 第一次质心对齐
        float[] alignedSignal = alignToCentroid(signal);

        // 查找在400到700索引范围内的局部最小值
        int minIndex = findLocalMinimumInRange(alignedSignal, 480, 520);

        if (minIndex != -1) {
            // 计算局部最小值与目标位置500的偏差
            int shiftAmount = TARGET_POSITION - minIndex;

            Log.d(TAG, "二次对齐: 局部最小值位置: " + minIndex);
            Log.d(TAG, "二次对齐: 偏移量: " + shiftAmount);

            // 进行二次平移
            float[] refinedSignal = new float[alignedSignal.length];
            for (int i = 0; i < alignedSignal.length; i++) {
                int shiftedIndex = (i + shiftAmount) % alignedSignal.length;
                if (shiftedIndex < 0) {
                    shiftedIndex += alignedSignal.length; // 处理负索引的情况
                }
                refinedSignal[shiftedIndex] = alignedSignal[i];
            }

            return refinedSignal;
        } else {
            // 如果在指定区间内没有找到局部最小值，直接返回第一次对齐的信号
            return alignedSignal;
        }
    }
}
