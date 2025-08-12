package com.example.watermonitor;

import android.graphics.Bitmap;

public class Result {
    public final float[] rgbSumValues;
    public final float[] r_values;
    public final float[] g_values;
    public final float[] b_values;
    public final Bitmap rgbBitmap;
    public final float[] rgbRealSumValues;

    public Result(float[] rgbSumValues, Bitmap rgbBitmap, float[] r_values, float[] g_values, float[] b_values, float[] rgbRealSumValues) {
        this.rgbSumValues = rgbSumValues;
        this.rgbBitmap = rgbBitmap;
        this.r_values = r_values;
        this.g_values = g_values;
        this.b_values = b_values;
        this.rgbRealSumValues = rgbRealSumValues;
    }
}
