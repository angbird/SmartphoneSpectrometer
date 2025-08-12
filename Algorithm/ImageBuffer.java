package com.example.watermonitor;

import java.util.LinkedList;
import java.util.Queue;

public class ImageBuffer {
    private final int capacity;
    private final int arrayLength;
    private final Queue<float[]> buffer;
    private final Object lock = new Object();

    public ImageBuffer(int capacity, int arrayLength) {
        this.capacity = capacity;
        this.arrayLength = arrayLength;
        this.buffer = new LinkedList<>();
    }

    public void add(float[] data) {
        synchronized (lock) {
            if (buffer.size() >= capacity) {
                buffer.poll(); // 移除队列头部（最早的数据）
            }
            buffer.add(data); // 添加新数据到队列尾部
        }
    }

    public float[] calculateAverage() {
        float[] average = new float[arrayLength];
        int size;

        synchronized (lock) {
            size = buffer.size();
            for (float[] array : buffer) {
                for (int i = 0; i < arrayLength; i++) {
                    average[i] += array[i];
                }
            }
        }

        if (size > 0) {
            for (int i = 0; i < arrayLength; i++) {
                average[i] /= size;
            }
        }

        return average;
    }

    public int getSize() {
        return buffer.size();
    }


    // 添加 clear 方法
    public void clear() {
        synchronized (lock) {
            buffer.clear();
        }
    }



    // 返回 buffer 的副本
    public float[][] getBufferContents() {
        synchronized (lock) {
            float[][] result = new float[buffer.size()][arrayLength];
            int index = 0;
            for (float[] array : buffer) {
                result[index++] = array.clone(); // 确保不会修改原始数据
            }
            return result;
        }
    }

}

