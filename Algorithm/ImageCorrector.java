package com.example.watermonitor;

import static com.example.watermonitor.PointSorter.sortPointsByAngleDescending;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Toast;

import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class ImageCorrector {

    private static final String TAG = "ImageCorrector";
    private Bitmap refImage;

    public ImageCorrector(Context context, String refImagePath) {
        // 加载参考图像
        refImage = loadReferenceImage(context, refImagePath);
        if (refImage == null) {
            Log.e(TAG, "参考图像加载失败！");
        } else {
            Log.d(TAG, "参考图像加载成功！");
        }
    }

    // 校正图像的具体实现
    public CorrectionResult correctImage(Context context, Bitmap bitmap) {
        Mat img2 = new Mat();
        Utils.bitmapToMat(bitmap, img2);
        Mat img2Copy = img2.clone();

        // 灰度化图像
        Mat img2_gray = new Mat();
        Imgproc.cvtColor(img2Copy, img2_gray, Imgproc.COLOR_BGR2GRAY);
//        logImage(context, "gray_image", img2_gray);

        // 二值化处理
        Mat img2_binary = new Mat();
        Imgproc.threshold(img2_gray, img2_binary, 10, 255, Imgproc.THRESH_BINARY);
//        logImage(context, "binary_image", img2_binary);

        // 闭运算处理
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(50, 50));
        Mat img2_closed = new Mat();
        Imgproc.morphologyEx(img2_binary, img2_closed, Imgproc.MORPH_CLOSE, kernel);
//        logImage(context, "closed_image", img2_closed);

        // 轮廓检测
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(img2_closed, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        Log.i(TAG, "在输入图像中找到 " + contours.size() + " 个轮廓。");

        // 设置参考图像的四个特征点
        Point[] pts1Array = new Point[]{
                new Point(1588, 233),
                new Point(712, 254),
                new Point(720, 620),
                new Point(1596, 600)
        };
        List<Point> pts1 = Arrays.asList(pts1Array);

        // 提取待校正图像的特征点 pts2
        List<Point> pts2 = new ArrayList<>();
        List<Point> rotated_pts2 = new ArrayList<>();
        // 在原图上绘制平滑后的最小外接矩形轮廓
        Mat imgWithContours = img2.clone();
        double  angle = 0;
        double minDeviation = 0;
        Point[] boxPoints = new Point[4];
        for (MatOfPoint contour : contours) {
            if (Imgproc.contourArea(contour) > 1000) {
                // 将轮廓转换为 MatOfPoint2f 类型，以便应用平滑处理
                MatOfPoint2f smoothContour = new MatOfPoint2f(contour.toArray());
                double epsilon = 0.01 * Imgproc.arcLength(smoothContour, true); // 控制平滑度的参数
                MatOfPoint2f approxContour = new MatOfPoint2f();
                // 应用平滑处理
                Imgproc.approxPolyDP(smoothContour, approxContour, epsilon, true);
                // 获取平滑后的最小外接矩形
                RotatedRect rect = Imgproc.minAreaRect(approxContour);
                // 获取旋转角度并记录日志
                angle = rect.angle;
                // 计算与水平和垂直的最小偏差
                minDeviation = Math.min(Math.abs(angle - 0), Math.abs(90 - angle));
                Log.d("最小外接矩形角度", "Angle: " + angle + "°");
                Log.d("最小外接矩形角度偏差", "minDeviation: " + minDeviation + "°");

                rect.points(boxPoints);
                // 将矩形点转换为 MatOfPoint 格式，便于绘制
                MatOfPoint matBoxPoints = new MatOfPoint(boxPoints);
                // 在图像上绘制平滑后的最小外接矩形
                List<MatOfPoint> boxContours = Arrays.asList(matBoxPoints);
                Imgproc.drawContours(imgWithContours, boxContours, -1, new Scalar(0, 255, 0), 2); // 使用绿色绘制矩形轮廓，粗细为 2
                // 将矩形角点添加到 pts2 作为特征点
                pts2.addAll(Arrays.asList(boxPoints));
            }
        }
        // 保存带有平滑后矩形轮廓的图像
//        logImage(context, "imgWithSmoothMinAreaRect", imgWithContours);

        // 确保至少检测到四个特征点
        Mat rotatedImage = new Mat();
        if (pts2.size() < 4) {
            Log.e(TAG, "特征点数量不足，无法进行校正");
            Toast.makeText(context, "校正失败：特征点不足，返回原图", Toast.LENGTH_SHORT).show();
            return new CorrectionResult(bitmap, pts2,0); // 返回原图和检测到的点
        }else { //图像旋转// 在此处进行图像旋转
            // 1. 获取旋转矩阵（逆时针旋转角度的负值，使得旋转角度归正为 0）
            Point center = new Point(img2.cols() / 2.0, img2.rows() / 2.0); // 图像中心点
            Mat rotationMatrix = Imgproc.getRotationMatrix2D(center, minDeviation, 1.0); // 使用负角度
            // 2. 旋转图像
            Imgproc.warpAffine(img2, rotatedImage, rotationMatrix, img2.size());
            // 3. 计算旋转后矩形角点的新坐标
            Point[] originalBoxPoints = boxPoints;

            Point[] rotatedBoxPoints = new Point[4];
            for (int i = 0; i < 4; i++) {
                double[] ptArr = new double[] {originalBoxPoints[i].x, originalBoxPoints[i].y, 1};
                Mat ptMat = new Mat(3, 1, CvType.CV_64F);
                ptMat.put(0, 0, ptArr);
                // 计算旋转后的点坐标
                Mat rotatedPtMat = new Mat();
                Core.gemm(rotationMatrix, ptMat, 1, new Mat(), 0, rotatedPtMat);
                rotatedBoxPoints[i] = new Point(rotatedPtMat.get(0, 0)[0], rotatedPtMat.get(1, 0)[0]);
            }
            // 将旋转后的矩形角点添加到 rotated_pts2 作为特征点
            rotated_pts2.addAll(Arrays.asList(rotatedBoxPoints));
            // 输出旋转后的角点坐标
//            for (int i = 0; i < 4; i++) {
//                Log.d("旋转后角点坐标", "Point " + i + ": (" + rotatedBoxPoints[i].x + ", " + rotatedBoxPoints[i].y + ")");
//            }
            // 如果需要，可以保存旋转后的图像
//             logImage(context, "RotatedImage", rotatedImage);
        }



        // 对 pts2 进行重新排序  按照角度排序
        List<Point> orderedPts2 = sortPointsByAngleDescending(pts2);
        List<Point> orderedRotatedPts2 = sortPointsByAngleDescending(rotated_pts2);
//        Log.i(TAG, "重新排序后的 pts2:");
//        for (Point pt : orderedPts2) {
//            Log.i(TAG, String.format("Point(x=%.2f, y=%.2f)", pt.x, pt.y));
//        }



//        // 只保留前三个检测到的特征点用于仿射变换
//        List<Point> pts1Subset = pts1.subList(0, 3);
//        List<Point> pts2Subset = orderedPts2.subList(0, 3);
//        // 转换为 MatOfPoint2f 格式
//        MatOfPoint2f pts1Mat = new MatOfPoint2f(pts1Subset.toArray(new Point[0]));
//        MatOfPoint2f pts2Mat = new MatOfPoint2f(pts2Subset.toArray(new Point[0]));
//        // 计算仿射变换矩阵
//        Mat transform = Imgproc.getAffineTransform(pts2Mat, pts1Mat);
//        if (transform.empty()) {
//            Log.e(TAG, "仿射变换矩阵生成失败");
//            Toast.makeText(context, "校正失败：无法生成变换矩阵，返回原图", Toast.LENGTH_SHORT).show();
//            return new CorrectionResult(bitmap, orderedPts2);
//        }
//        Mat alignedMat = new Mat();
//        Imgproc.warpAffine(img2, alignedMat, transform, img2.size());
//        // 转换校正后的 Mat 到 Bitmap
//        Bitmap alignedBitmap = Bitmap.createBitmap(alignedMat.cols(), alignedMat.rows(), Bitmap.Config.ARGB_8888);
//        Utils.matToBitmap(alignedMat, alignedBitmap);
////        logImage(context, "corrected_image", alignedMat);

        // 转换旋转后的 Mat 到 Bitmap
        Bitmap rotatedBitmap = Bitmap.createBitmap(rotatedImage.cols(), rotatedImage.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rotatedImage, rotatedBitmap);
//        logImage(context, "rotated_image", rotatedImage);

        // 释放资源
//        pts1Mat.release();
//        pts2Mat.release();
//        transform.release();
        img2.release();
        img2_gray.release();
        img2_binary.release();
        img2_closed.release();
        rotatedImage.release();
//        alignedMat.release();

        return new CorrectionResult(rotatedBitmap, orderedRotatedPts2, minDeviation);
    }


    // 保存和记录处理步骤的中间图像
    private void logImage(Context context, String imageName, Mat image) {
        // 将 Mat 转换为 Bitmap
        Bitmap bmp = Bitmap.createBitmap(image.width(), image.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(image, bmp);

        // 生成带时间戳的唯一文件名
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        String imageFileName = imageName + "_" + timeStamp + ".jpg";

        // 准备 ContentValues
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, imageFileName);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/WaterMonitorImages");

        // 获取 ContentResolver
        ContentResolver resolver = context.getContentResolver();

        // 将图像插入到 MediaStore 中
        try {
            Uri uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
            if (uri != null) {
                OutputStream outputStream = resolver.openOutputStream(uri);
                if (outputStream != null) {
                    bmp.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);
                    outputStream.close();
                    Log.i(TAG, "图像保存成功: " + imageFileName);
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "图像保存失败: " + e.getMessage());
        }
    }


    // 加载参考图像的方法
    private Bitmap loadReferenceImage(Context context, String refImagePath) {
        try {
            InputStream inputStream = context.getAssets().open(refImagePath);
            return BitmapFactory.decodeStream(inputStream);
        } catch (IOException e) {
            Log.e(TAG, "Error loading reference image: ", e);
            return null;
        }
    }
}
