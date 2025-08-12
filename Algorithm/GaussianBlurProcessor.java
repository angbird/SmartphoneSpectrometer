package com.example.watermonitor;

import android.graphics.Bitmap;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class GaussianBlurProcessor {

    /**
     * 对给定的 Bitmap 图像应用高斯滤波
     *
     * @param bitmap    输入的 Bitmap 图像
     * @param kernelSize 滤波核大小，必须为奇数（如 3, 5, 7）
     * @return 高斯滤波处理后的 Bitmap 图像
     */
    public static Bitmap applyGaussianBlur(Bitmap bitmap, int kernelSize) {
        if (kernelSize % 2 == 0) {
            throw new IllegalArgumentException("Kernel size must be an odd number");
        }

        // 将 Bitmap 转换为 OpenCV Mat 格式
        Mat srcMat = new Mat();
        Utils.bitmapToMat(bitmap, srcMat);

        // 创建目标 Mat 用于存储滤波结果
        Mat destMat = new Mat(srcMat.rows(), srcMat.cols(), CvType.CV_8UC3);

        // 应用高斯滤波
        Size size = new Size(kernelSize, kernelSize);
        Imgproc.GaussianBlur(srcMat, destMat, size, 0);

        // 将结果转换回 Bitmap
        Bitmap blurredBitmap = Bitmap.createBitmap(destMat.cols(), destMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(destMat, blurredBitmap);

        // 释放资源
        srcMat.release();
        destMat.release();

        return blurredBitmap;
    }
}
