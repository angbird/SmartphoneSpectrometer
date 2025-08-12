package com.example.watermonitor;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;

import java.util.List;

public class AffineTransformHelper {
    private static final String TAG = "AffineTransformHelper";

    public static Mat calculateAffineTransform(List<Point> averagePoints, List<Point> refFeaturepoints) {
        // 确保输入点列表的大小至少为3，以便进行仿射变换
        if (averagePoints.size() < 3 || refFeaturepoints.size() < 3) {
            throw new IllegalArgumentException("Both point lists must contain at least 3 points for affine transformation.");
        }

        // 取前三个点构建仿射变换
        List<Point> averagePointsSubset = averagePoints.subList(0, 3);
        List<Point> refFeaturepointsSubset = refFeaturepoints.subList(0, 3);

        // 转换为 MatOfPoint2f 格式
        MatOfPoint2f srcPoints = new MatOfPoint2f();
        srcPoints.fromList(averagePointsSubset);

        MatOfPoint2f dstPoints = new MatOfPoint2f();
        dstPoints.fromList(refFeaturepointsSubset);

        // 计算仿射变换矩阵
        Mat transform = Imgproc.getAffineTransform(srcPoints, dstPoints);

        // 释放资源
        srcPoints.release();
        dstPoints.release();

        return transform;
    }


    // 打印仿射变换矩阵
    public static void logAffineTransformMatrix(Mat transform) {
        if (transform.empty()) {
            Log.e(TAG, "Transform matrix is empty.");
            return;
        }

        double[] data = new double[(int) (transform.total() * transform.channels())];
        transform.get(0, 0, data);

        StringBuilder matrixString = new StringBuilder("Affine Transform Matrix:\n");
        for (int i = 0; i < transform.rows(); i++) {
            for (int j = 0; j < transform.cols(); j++) {
                matrixString.append(String.format("%.2f", data[i * transform.cols() + j])).append(" ");
            }
            matrixString.append("\n");
        }
        Log.d(TAG, matrixString.toString());
    }




}

