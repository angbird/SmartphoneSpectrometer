package com.example.watermonitor;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Base64;

import androidx.room.TypeConverter;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

public class Converters {
    @TypeConverter
    public static String fromFloatArray(float[] floats) {
        if (floats == null) {
            return null;
        }
        StringBuilder sb = new StringBuilder();
        for (float value : floats) {
            sb.append(value).append(",");
        }
        // 去掉最后一个多余的逗号
        if (sb.length() > 0) {
            sb.setLength(sb.length() - 1);
        }
        return sb.toString();
    }

    @TypeConverter
    public static float[] toFloatArray(String data) {
        if (data == null || data.isEmpty()) {
            return new float[0];
        }
        String[] parts = data.split(",");
        float[] floats = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            floats[i] = Float.parseFloat(parts[i]);
        }
        return floats;
    }


    @TypeConverter
    public static String fromImageBufferList(List<float[]> imageBufferList) {
        // 使用线程安全的 CopyOnWriteArrayList 避免并发修改异常
        List<float[]> safeList = new CopyOnWriteArrayList<>(imageBufferList);
        StringBuilder sb = new StringBuilder();
        for (float[] array : safeList) {
            for (int i = 0; i < array.length; i++) {
                sb.append(array[i]);
                if (i < array.length - 1) {
                    sb.append(",");
                }
            }
            sb.append(";");  // 每个数组用分号分隔
        }
        // 去掉最后一个多余的分号
        if (sb.length() > 0) {
            sb.setLength(sb.length() - 1);
        }
        return sb.toString();
    }

    @TypeConverter
    public static List<float[]> toImageBufferList(String data) {
        if (data == null || data.isEmpty()) {
            // 返回一个空的列表或根据需求处理
            return new ArrayList<>();
        }

        List<float[]> imageBufferList = new ArrayList<>();
        String[] arrays = data.split(";");
        for (String array : arrays) {
            String[] values = array.split(",");
            float[] floatArray = new float[values.length];
            for (int i = 0; i < values.length; i++) {
                floatArray[i] = Float.parseFloat(values[i]);
            }
            imageBufferList.add(floatArray);
        }
        return imageBufferList;
    }

    @TypeConverter
    public static String fromBitmap(Bitmap bitmap) {
        if (bitmap == null) return null;
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream);
        byte[] byteArray = outputStream.toByteArray();
        return Base64.encodeToString(byteArray, Base64.DEFAULT);
    }

    @TypeConverter
    public static Bitmap toBitmap(String encodedString) {
        if (encodedString == null) return null;
        byte[] decodedBytes = Base64.decode(encodedString, Base64.DEFAULT);
        return BitmapFactory.decodeByteArray(decodedBytes, 0, decodedBytes.length);
    }





}

