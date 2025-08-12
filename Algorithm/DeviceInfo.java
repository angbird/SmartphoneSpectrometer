package com.example.watermonitor;

import android.os.Build;
import android.util.Log;

public class DeviceInfo {
    private static final String TAG = "手机型号信息";
    public static void logDeviceInfo() {
        // 手机制造商
        String manufacturer = Build.MANUFACTURER;

        // 手机型号
        String model = Build.MODEL;

        // 系统版本号
        String version = Build.VERSION.RELEASE;

        // 打印信息
        Log.d(TAG, "制造商: " + manufacturer);
        Log.d(TAG, "型号: " + model);
        Log.d(TAG, "安卓版本: " + version);

    }
}

