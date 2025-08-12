package com.example.watermonitor;

import android.util.Log;

import java.util.ArrayList;
import java.util.List;

public class CrossCorrelation {
    private static final String TAG = "CrossCorrelation";
    private float[] noSampleSpec; // 参考信号

    // 构造函数，初始化参考信号
    public CrossCorrelation(float[] noSampleSpec) {
        this.noSampleSpec = noSampleSpec;
    }

    // 互相关计算方法
    public float[] compensate(float[] sampleSpec) {
        int len = sampleSpec.length;
        float[] correlation = new float[2 * len - 1];

        // 计算互相关
        for (int i = 0; i < correlation.length; i++) {
            int shift = i - (len - 1);
            float sum = 0;
            for (int j = 0; j < len; j++) {
                int k = j + shift;
                if (k >= 0 && k < len) {
                    sum += sampleSpec[j] * noSampleSpec[k];
                }
            }
            correlation[i] = sum;
        }

        // 找到最大相关性的位置
        int maxIndex = 0;
        for (int i = 1; i < correlation.length; i++) {
            if (correlation[i] > correlation[maxIndex]) {
                maxIndex = i;
            }
        }

        // 计算信号的补偿量
        int shiftAmount =  maxIndex - (len - 1);
        Log.d(TAG, "偏移大小: " + shiftAmount); // 记录偏移大小
        float[] compensatedSignal = new float[len];
        for (int i = 0; i < len; i++) {
            int index = i - shiftAmount;
            if (index >= 0 && index < len) {
                compensatedSignal[i] = sampleSpec[index];
            } else {
                compensatedSignal[i] = 0; // 边界处理
            }
        }

        return compensatedSignal;
    }










}

