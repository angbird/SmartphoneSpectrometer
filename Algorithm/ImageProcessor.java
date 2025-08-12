package com.example.watermonitor;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.YuvImage;
import android.media.Image;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import androidx.annotation.NonNull;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;


public class ImageProcessor {

    private static final String TAG = "ImageProcessor";
    private static final String SERVER_URL = "http://47.108.141.140:8080/api/image/correct"; // 替换为服务器URL
    private static final int START_ROW = 951;
    private static final int END_ROW = 1250;
//    private static final int START_ROW = 10;
//    private static final int END_ROW = 1900;

    private static final int IMAGE_WIDTH = 1920;
    private static final int IMAGE_HEIGHT = 1080;
    rgbchannel rgb = new rgbchannel();
    private double[] R_luminanceValues = rgb.getRValues();
    private double[] G_luminanceValues = rgb.getGValues();
    private double[] B_luminanceValues = rgb.getBValues();
    // ImageProcessor 类的构造方法，这里不需要参数，因为它只是一个简单的工具类

    // 定义 processImage 方法来处理 Image 对象
    public interface TaskListener {
        void onTaskCompleted(Result result);
    }


    public void processImageAsync(Bitmap bitmap, TaskListener listener) {
        new ProcessImageTask(listener).execute(bitmap);
    }
    private class ProcessImageTask extends AsyncTask<Bitmap, Void, Result> {
        private final TaskListener listener;

        public ProcessImageTask(TaskListener listener) {
            this.listener = listener;
        }
        @Override
        protected Result doInBackground(Bitmap... bitmaps) {
            if (bitmaps.length == 0) {
                return null;
            }
            Bitmap bitmap = bitmaps[0];
            try {
                return processImage(bitmap);
            } catch (IOException e) {
                e.printStackTrace();
                return null;
            }
        }

        @Override
        protected void onPostExecute(Result result) {
            super.onPostExecute(result);
            if (listener != null) {
                listener.onTaskCompleted(result);
            }
        }

        private Result processImage(Bitmap bitmap) throws IOException {
            float[][] rgb_and_sum_values;
            float[] rgb_sum_values = new float[0];
            float[] r_values = new float[0];
            float[] g_values = new float[0];
            float[] b_values = new float[0];
            float[] rgb_real_sum_values = new float[0];
            if (bitmap != null) {
                if (bitmap.getWidth() != IMAGE_WIDTH || bitmap.getHeight() != IMAGE_HEIGHT) {
                    Log.e(TAG, "Unexpected image size");
                    return null;
                }

                rgb_and_sum_values = calculate_rgb_sum_with_opencv(bitmap);  //图像压缩为一维数据
                r_values = rgb_and_sum_values[0];
                g_values = rgb_and_sum_values[1];
                b_values = rgb_and_sum_values[2];
                rgb_sum_values = rgb_and_sum_values[3];
                rgb_real_sum_values = rgb_and_sum_values[4];

                return new Result(rgb_sum_values, bitmap,r_values,g_values,b_values,rgb_real_sum_values);
            }
           return null;
        }



    }


    //将bitmap转换为字节数组
    private byte[] bitmapToByteArray(Bitmap bitmap) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, baos); // 将Bitmap压缩成JPEG格式
        return baos.toByteArray();
    }




    /*原始方法对图像进行处理*/
    private float[][] calculate_rgb_sum(Bitmap bitmap) {
        // 初始化存储R、G、B通道值的二维数组
        int width = bitmap.getHeight();
        int height = bitmap.getWidth();
//        Log.d(TAG, "calculate_rgb_sum: 高度"+height);
//        Log.d(TAG, "calculate_rgb_sum: 宽度"+width);
        int[][] rChannelValues = new int[END_ROW - START_ROW + 1][width];
        int[][] rChannelFilteredValues;
        int[][] gChannelValues = new int[END_ROW - START_ROW + 1][width];
        int[][] gChannelFilteredValues;
        int[][] bChannelValues = new int[END_ROW - START_ROW + 1][width];
        int[][] bChannelFilteredValues;
        float[] r_values;
        float[] g_values;
        float[] b_values;
        float[] r_RealValues;
        float[] g_RealValues;
        float[] b_RealValues;


        // 遍历指定行范围
        for (int row = START_ROW; row <= END_ROW; row++) {
            // 确保不读取超出数组边界的数据
            if (row >= height) {
                Log.e(TAG, "Image data is not large enough for the given row");
                break;
            }

            // 遍历当前行的每个像素
            for (int col = 0; col < width; col++) {
                // 获取像素值
                int pixel = bitmap.getPixel(row,col);
                //存储位置  正常是反的
                int position = width-1-col;

                // 读取R、G、B通道的值
                int rValue = (pixel >> 16) & 0xFF;
                int gValue = (pixel >> 8) & 0xFF;
                int bValue = pixel & 0xFF;

                // 存储到对应的二维数组中
                rChannelValues[row - START_ROW][position] = rValue;
                gChannelValues[row - START_ROW][position] = gValue;
                bChannelValues[row - START_ROW][position] = bValue;
            }
        }

        // 进行滤波操作
        rChannelFilteredValues = applyMeanFilter(rChannelValues, 3);
        gChannelFilteredValues = applyMeanFilter(gChannelValues, 3);
        bChannelFilteredValues = applyMeanFilter(bChannelValues, 3);

        // 计算每列的平均值
        r_values = averageEachColumn(rChannelFilteredValues);
        g_values = averageEachColumn(gChannelFilteredValues);
        b_values = averageEachColumn(bChannelFilteredValues);

        // 创建一个新的数组来存储映射后的光度值
        r_RealValues = new float[r_values.length];
        g_RealValues = new float[g_values.length];
        b_RealValues = new float[b_values.length];
        // 将r_values中的每个值映射为光度值
        for (int i = 0; i < r_values.length; i++) {
            int pixelValue_r = (int) r_values[i];
            int pixelValue_g = (int) g_values[i];
            int pixelValue_b = (int) b_values[i];
            if (pixelValue_r >= 0 && pixelValue_r <= 255) {
                r_RealValues[i] = (float) R_luminanceValues[pixelValue_r];
                g_RealValues[i] = (float) G_luminanceValues[pixelValue_g];
                b_RealValues[i] = (float) B_luminanceValues[pixelValue_b];
//                Log.d(TAG, "calculate_rgb_sum: 值"+luminanceMappedValues[i]);
            } else {
                Log.e(TAG, "Pixel value out of range: " + pixelValue_r);
            }
        }



        // 计算RGB通道的加和值

        float[] temp1 = addArrays(r_RealValues, g_RealValues, b_RealValues);
        float[] rgbRealSumValues = RemoveNoise(temp1,5);

        // 滤波 加和
        float[] temp2 = addArrays(r_values, g_values, b_values);
        float[] rgbSumValues = RemoveNoise(temp2,5);

        // 返回封装好的结果
        return new float[][]{r_values,g_values,b_values,rgbSumValues,rgbRealSumValues};
    }

    //去噪
    private float[] RemoveNoise(float[] rgbSumValues, int windowSize) {
        // 边界条件检查
        if (windowSize <= 0 || rgbSumValues == null || rgbSumValues.length < windowSize) {
            throw new IllegalArgumentException("Invalid window size or input array");
        }

        float[] smoothedValues = new float[rgbSumValues.length];
        int halfWindow = windowSize / 2;

        // 应用移动平均法
        for (int i = 0; i < rgbSumValues.length; i++) {
            float sum = 0;
            int count = 0;

            // 窗口范围从 i - halfWindow 到 i + halfWindow
            for (int j = -halfWindow; j <= halfWindow; j++) {
                int index = i + j;
                // 检查索引是否在有效范围内
                if (index >= 0 && index < rgbSumValues.length) {
                    sum += rgbSumValues[index];
                    count++;
                }
            }

            // 计算平均值
            smoothedValues[i] = sum / count;
        }

        return smoothedValues;
    }

    public static int[][] applyMeanFilter(int[][] ChannelValues, int filterSize) {
        if (filterSize % 2 == 0) {
            throw new IllegalArgumentException("Filter size must be odd.");
        }
        int halfSize = filterSize / 2;
        int height = ChannelValues.length;
        int width = ChannelValues[0].length;

        // 创建一个新数组来存储滤波后的值
        int[][] filteredValues = new int[height][width];

        for (int row = halfSize; row < height - halfSize; row++) {
            for (int col = halfSize; col < width - halfSize; col++) {
                int sum = 0;
                for (int i = -halfSize; i <= halfSize; i++) {
                    for (int j = -halfSize; j <= halfSize; j++) {
                        // 注意：这里需要处理边界条件，确保索引不会越界
                        int newRow = row - i;
                        int newCol = col - j;
                        if (newRow >= 0 && newRow < height && newCol >= 0 && newCol < width) {
                            sum += ChannelValues[newRow][newCol];
                        }
                    }
                }
                filteredValues[row - halfSize][col - halfSize] = sum / (filterSize * filterSize);
            }
        }

        // 边缘像素未被处理，如果需要，可以添加额外的逻辑来处理它们

        // 返回滤波后的数组
        return filteredValues;
    }

    public static float[] averageEachColumn(int[][] rChannelFilteredValues) {
        if (rChannelFilteredValues == null || rChannelFilteredValues.length == 0 || rChannelFilteredValues[0].length == 0) {
            throw new IllegalArgumentException("The matrix must have at least one row and one column.");
        }
        int width = rChannelFilteredValues[0].length; // 列数（假设所有行都有相同的列数）
        float[] columnAverages = new float[width];

        for (int col = 0; col < width; col++) {
            int sum = 0;
            for (int row = 0; row < rChannelFilteredValues.length; row++) {
                sum += rChannelFilteredValues[row][col];
            }
            columnAverages[col] = (float) sum / rChannelFilteredValues.length;
        }

        return columnAverages;
    }

    public static float[] addArrays(float[] array1, float[] array2, float[] array3) {
        if (array1 == null || array2 == null || array3 == null) {
            throw new IllegalArgumentException("All arrays must not be null");
        }

        if (array1.length != array2.length || array2.length != array3.length) {
            throw new IllegalArgumentException("All arrays must have the same length");
        }

        int length = array1.length;
        float[] result = new float[length];

        for (int i = 0; i < length; i++) {
//            result[i] = (float) (0.3*array1[i] + 0.5*array2[i] + 0.2*array3[i]);
            result[i] = (float) (0.48*array1[i] + 0.48*array2[i] + 0.48*array3[i]);
        }

        return result;
    }









    /*利用opencv对图像进行处理*/

    public float[][] calculate_rgb_sum_with_opencv(Bitmap bitmap) {
        // 检查 Bitmap 是否有效
        if (bitmap == null || bitmap.getWidth() == 0 || bitmap.getHeight() == 0) {
            Log.e("calculate_rgb_sum", "Invalid Bitmap input");
            return null;
        }
        // 1. 旋转 Bitmap 顺时针 90°
        Matrix matrix = new Matrix();
        matrix.postRotate(90);
        Bitmap rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

        // 2. 转换旋转后的 Bitmap 到 OpenCV 的 Mat
        Mat mat = new Mat();
        Utils.bitmapToMat(rotatedBitmap, mat);

        // 检查指定行范围是否合法
        if (START_ROW < 0 || END_ROW >= mat.rows()) {
            Log.e(TAG, "Row range out of bounds: START_ROW=" + START_ROW + ", END_ROW=" + END_ROW + ", Mat rows=" + mat.rows());
            return null;
        }
//        Log.e(TAG, "Row range out of bounds: START_ROW=" + START_ROW + ", END_ROW=" + END_ROW + ", Mat rows=" + mat.rows());

        // 提取感兴趣区域 (ROI)
        org.opencv.core.Rect roi = new Rect(0, START_ROW, mat.cols(), END_ROW - START_ROW + 1);
        Mat roiMat = mat.submat(roi);
        Log.d(TAG, "roiMat size: " + roiMat.size());


        // 分离通道
        List<Mat> channels = new ArrayList<>();
        Core.split(roiMat, channels);
        Mat rChannel = channels.get(2); // OpenCV 中，顺序是 BGR
        Mat gChannel = channels.get(1);
        Mat bChannel = channels.get(0);


//        Log.d(TAG, "rChannel size: " + rChannel.size());
//        Log.d(TAG, "gChannel size: " + gChannel.size());
//        Log.d(TAG, "bChannel size: " + bChannel.size());
//
//        Log.d(TAG, "Mean of R Channel: " + Core.mean(rChannel).val[0]);
//        Log.d(TAG, "Mean of G Channel: " + Core.mean(gChannel).val[0]);
//        Log.d(TAG, "Mean of B Channel: " + Core.mean(bChannel).val[0]);


        // 滤波处理
        Mat rFiltered = new Mat();
        Mat gFiltered = new Mat();
        Mat bFiltered = new Mat();
        Imgproc.blur(rChannel, rFiltered, new Size(3, 3)); // 均值滤波
        Imgproc.blur(gChannel, gFiltered, new Size(3, 3));
        Imgproc.blur(bChannel, bFiltered, new Size(3, 3));

        // 将 rFiltered、gFiltered 和 bFiltered 转换为浮点类型，确保类型一致
        Mat rFilteredFloat = new Mat();
        Mat gFilteredFloat = new Mat();
        Mat bFilteredFloat = new Mat();
        rFiltered.convertTo(rFilteredFloat, CvType.CV_32F);
        gFiltered.convertTo(gFilteredFloat, CvType.CV_32F);
        bFiltered.convertTo(bFilteredFloat, CvType.CV_32F);

        // 计算每列的平均值
        Mat rColumnMeans = new Mat();
        Mat gColumnMeans = new Mat();
        Mat bColumnMeans = new Mat();
        Core.reduce(rFilteredFloat, rColumnMeans, 0, Core.REDUCE_AVG); // 按列计算平均值
        Core.reduce(gFilteredFloat, gColumnMeans, 0, Core.REDUCE_AVG);
        Core.reduce(bFilteredFloat, bColumnMeans, 0, Core.REDUCE_AVG);

        // 检查 rColumnMeans 的类型和大小
//        Log.d(TAG, "rColumnMeans size: " + rColumnMeans.size());
//        Log.d(TAG, "rColumnMeans type: " + CvType.typeToString(rColumnMeans.type()));

        // 提取 rColumnMeans 数据到数组
        float[] r_values = new float[(int) rColumnMeans.total()];
        rColumnMeans.get(0, 0, r_values);

        // 提取 gColumnMeans 数据到数组
        float[] g_values = new float[(int) gColumnMeans.total()];
        gColumnMeans.get(0, 0, g_values);

        // 提取 bColumnMeans 数据到数组
        float[] b_values = new float[(int) bColumnMeans.total()];
        bColumnMeans.get(0, 0, b_values);

//        // 打印 r_values 到日志
//        StringBuilder rValuesLog = new StringBuilder("r_values: [");
//        for (int i = 0; i < r_values.length; i++) {
//            rValuesLog.append(String.format("%.2f", r_values[i]));
//            if (i < r_values.length - 1) {
//                rValuesLog.append(", ");
//            }
//        }
//        rValuesLog.append("]");
//        Log.d(TAG, rValuesLog.toString());
//
//        // 打印 g_values 到日志
//        StringBuilder gValuesLog = new StringBuilder("g_values: [");
//        for (int i = 0; i < g_values.length; i++) {
//            gValuesLog.append(String.format("%.2f", g_values[i]));
//            if (i < g_values.length - 1) {
//                gValuesLog.append(", ");
//            }
//        }
//        gValuesLog.append("]");
//        Log.d(TAG, gValuesLog.toString());
//
//        // 打印 b_values 到日志
//        StringBuilder bValuesLog = new StringBuilder("b_values: [");
//        for (int i = 0; i < b_values.length; i++) {
//            bValuesLog.append(String.format("%.2f", b_values[i]));
//            if (i < b_values.length - 1) {
//                bValuesLog.append(", ");
//            }
//        }
//        bValuesLog.append("]");
//        Log.d(TAG, bValuesLog.toString());


        // 将平均值映射为光度值
        // 使用 Arrays.setAll 一次性修改数组中的所有元素

        float[] r_RealValues = mapToLuminance(r_values, convertDoubleToFloatArray(R_luminanceValues),1f);
        float[] g_RealValues = mapToLuminance(g_values, convertDoubleToFloatArray(G_luminanceValues),1f);
        float[] b_RealValues = mapToLuminance(b_values, convertDoubleToFloatArray(B_luminanceValues),1f);

        // 计算 RGB 通道的加和
        float[] rgbRealSumValues = addArraysAndRemoveNoise(r_RealValues, g_RealValues, b_RealValues, 5);
        float[] rgbSumValues = addArraysAndRemoveNoise(r_values, g_values, b_values, 5);

//        // 将数组的内容打印到日志
//        if (rgbSumValues != null) {
//            StringBuilder rgbSumValuesBuilder = new StringBuilder();
//            rgbSumValuesBuilder.append("rgbSumValues: [");
//            for (int i = 0; i < rgbSumValues.length; i++) {
//                rgbSumValuesBuilder.append(String.format("%.2f", rgbSumValues[i]));
//                if (i < rgbSumValues.length - 1) {
//                    rgbSumValuesBuilder.append(", "); // 添加逗号分隔符
//                }
//            }
//            rgbSumValuesBuilder.append("]");
//            Log.d("RGB_LOG", rgbSumValuesBuilder.toString());
//        } else {
//            Log.d("RGB_LOG", "rgbSumValues is null");
//        }

        // 返回结果
        return new float[][]{r_values, g_values, b_values, rgbSumValues, rgbRealSumValues};
    }

    // 将颜色值映射到光度值
    private float[] mapToLuminance(float[] values, float[] luminanceTable, float coefficient) {
        float[] result = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            int value = Math.round(values[i] * coefficient);
            if (value >= 0 && value < luminanceTable.length) {
                result[i] = luminanceTable[value];
            } else {
                result[i] = 0; // 默认值
            }
        }
        return result;
    }

    // 加和多个数组并降噪
    private float[] addArraysAndRemoveNoise(float[] r, float[] g, float[] b, int filterSize) {
        float[] sum = new float[r.length];
        for (int i = 0; i < r.length; i++) {
            sum[i] = (float) (0.48*(r[i] + g[i] + b[i]));
//            sum[i] = Math.max(r[i], Math.max(g[i], b[i]));
//            Log.d(TAG, "addArraysAndRemoveNoise: 最大值");
        }
        return removeNoise(sum, filterSize);
    }

    // 降噪处理 (简单的均值滤波)
    private float[] removeNoise(float[] values, int filterSize) {
        float[] smoothed = new float[values.length];
        int halfSize = filterSize / 2;
        for (int i = 0; i < values.length; i++) {
            float sum = 0;
            int count = 0;
            for (int j = -halfSize; j <= halfSize; j++) {
                int index = i + j;
                if (index >= 0 && index < values.length) {
                    sum += values[index];
                    count++;
                }
            }
            smoothed[i] = sum / count;
        }
        return smoothed;
    }
    //double数组转为float数组
    private static float[] convertDoubleToFloatArray(double[] doubleArray) {
        float[] floatArray = new float[doubleArray.length];
        for (int i = 0; i < doubleArray.length; i++) {
            floatArray[i] = (float) doubleArray[i];
        }
        return floatArray;
    }







}

