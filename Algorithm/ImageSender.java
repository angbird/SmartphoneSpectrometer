package com.example.watermonitor;

import android.graphics.Bitmap;

import okhttp3.*;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import org.json.JSONArray;
import org.json.JSONObject;

public class ImageSender {

    public interface ResponseCallback {
        void onSuccess(JSONArray compressedSignals);
        void onError(String errorMessage);
    }

    public void sendBitmapToServer(Bitmap bitmap, String serverUrl, ResponseCallback callback) {
        new Thread(() -> {
            try {
                // 将 Bitmap 转换为字节数组
                ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
                byte[] imageBytes = byteArrayOutputStream.toByteArray();

                // 使用 OkHttp 构造 Multipart 请求
                OkHttpClient client = new OkHttpClient();
                RequestBody requestBody = new MultipartBody.Builder()
                        .setType(MultipartBody.FORM)
                        .addFormDataPart(
                                "images",
                                "image.jpg",
                                RequestBody.create(imageBytes, MediaType.parse("image/jpeg"))
                        )
                        .build();

                Request request = new Request.Builder()
                        .url(serverUrl)
                        .post(requestBody)
                        .build();

                // 发送请求并处理响应
                try (Response response = client.newCall(request).execute()) {
                    if (response.isSuccessful()) {
                        String responseBody = response.body().string();
                        JSONObject jsonResponse = new JSONObject(responseBody); // 使用 JSONObject 解析响应
                        JSONArray compressedSignals = jsonResponse.getJSONArray("compressed_signals"); // 提取 compressed_signals 数组
                        callback.onSuccess(compressedSignals);
                    } else {
                        callback.onError("发送失败，响应码: " + response.code());
                    }
                }

            } catch (Exception e) {
                callback.onError("发送时出现异常: " + e.getMessage());
            }
        }).start();
    }
}
