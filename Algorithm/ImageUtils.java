package com.example.watermonitor;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Base64;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

public class ImageUtils {

    private static final String TAG = "ImageUtils";

    // 将 YUV_420_888 格式的 Image 转换为 NV21 格式的字节数组
    public static byte[] YUV_420_888toNV21(Image image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int ySize = width * height;
        int uvSize = width * height / 4;

        byte[] nv21 = new byte[ySize + uvSize * 2];

        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();

        int rowStride = image.getPlanes()[0].getRowStride();
        int pos = 0;

        // Y通道
        for (int i = 0; i < height; i++) {
            yBuffer.get(nv21, pos, width);
            pos += width;
            if (i < height - 1) {
                yBuffer.position(yBuffer.position() + rowStride - width);
            }
        }

        // U和V通道
        rowStride = image.getPlanes()[2].getRowStride();
        int pixelStride = image.getPlanes()[2].getPixelStride();

        for (int i = 0; i < height / 2; i++) {
            for (int j = 0; j < width / 2; j++) {
                nv21[pos++] = vBuffer.get(j * pixelStride + i * rowStride);
                nv21[pos++] = uBuffer.get(j * pixelStride + i * rowStride);
            }
        }

        return nv21;
    }

    // 将 NV21 格式的字节数组转换为 Bitmap
    public static Bitmap nv21ToBitmap(byte[] nv21, int width, int height) {
        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, width, height, null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, width, height), 100, out);
        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    // 将 YUV_420_888 格式的 Image 转换为 Bitmap
    public static Bitmap YUV_420_888toBitmap(Image image) {
        byte[] nv21 = YUV_420_888toNV21(image);
        return nv21ToBitmap(nv21, image.getWidth(), image.getHeight());
    }

    // 执行仿射变换的函数
    public static Bitmap applyAffineTransform(Bitmap inputBitmap, Mat transform) {
        // 将 Bitmap 转换为 Mat
        Mat srcMat = new Mat();
        Utils.bitmapToMat(inputBitmap, srcMat);

        // 创建一个 Mat 来存储输出
        Mat dstMat = new Mat();

        // 获取输入 Bitmap 的尺寸
        Size size = new Size(inputBitmap.getWidth(), inputBitmap.getHeight());

        // 对输入图像进行仿射变换
        Imgproc.warpAffine(srcMat, dstMat, transform, size);

        // 将变换后的 Mat 转回 Bitmap
        Bitmap outputBitmap = Bitmap.createBitmap(dstMat.cols(), dstMat.rows(), inputBitmap.getConfig());
        Utils.matToBitmap(dstMat, outputBitmap);

        // 释放资源
        srcMat.release();
        dstMat.release();

        return outputBitmap;
    }



    public static Bitmap rotateBitmap(Bitmap bitmap, double angle) {
        if (bitmap == null) {
            throw new IllegalArgumentException("Bitmap is null.");
        }
        // 将 Bitmap 转换为 Mat
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap, mat);
        // 获取图像的中心点
        Point center = new Point(mat.cols() / 2.0, mat.rows() / 2.0);
        // 创建旋转矩阵
        Mat rotationMatrix = Imgproc.getRotationMatrix2D(center, angle, 1.0);
        // 创建旋转后的 Mat 容器
        Mat rotatedMat = new Mat();
        // 旋转图像
        Imgproc.warpAffine(mat, rotatedMat, rotationMatrix, mat.size());
        // 将旋转后的 Mat 转换回 Bitmap
        Bitmap rotatedBitmap = Bitmap.createBitmap(rotatedMat.cols(), rotatedMat.rows(), bitmap.getConfig());
        Utils.matToBitmap(rotatedMat, rotatedBitmap);
        return rotatedBitmap;
    }




    // 将 Bitmap 转换为 Base64 字符串
    public static String bitmapToBase64(Bitmap bitmap) throws IOException {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream);  // 或使用其他格式
        byte[] byteArray = byteArrayOutputStream.toByteArray();
        byteArrayOutputStream.close();
        return Base64.encodeToString(byteArray, Base64.DEFAULT);
    }






}

