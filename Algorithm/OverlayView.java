package com.example.watermonitor;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;

public class OverlayView extends View {
    private Paint paint;
    private int top, bottom, left, right;

    public OverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        paint = new Paint();
        paint.setColor(Color.RED);  // 设置矩形框的颜色
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(5);    // 设置矩形框的边框宽度


        // 初始化矩形框位置
        top = 950;
        bottom = 1250;
        left = 380;
        right = 680;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        // 获取视图的宽度和高度
        int viewWidth = getWidth();
        int viewHeight = getHeight();
        // 图像的实际宽高比（16:9），即 1920:1080
        float aspectRatio = 1920f / 1080f;

        Log.d("视图宽", String.valueOf(viewWidth));
        Log.d("视图高", String.valueOf(viewHeight));


        int rectTop = top * viewHeight / 1920;  // 将图像宽度比例换算到视图宽度
        int rectBottom = bottom * viewHeight / 1920;
        int rectLeft = left * viewWidth / 1920 ;  // 矩形框从顶部开始
        int rectRight = right * viewWidth / 1920;  // 矩形框到底部结束

        Log.d("上下左右", "上"+String.valueOf(rectTop)+"下："+String.valueOf(rectBottom)+"左："+String.valueOf(rectLeft)+"右:"+String.valueOf(rectRight));

        // 设置矩形框的范围
//        int rectTop = top * viewHeight / 1920;  // 将图像宽度比例换算到视图宽度
//        int rectBottom = bottom * viewHeight / 1920;
//        int rectLeft = left * viewWidth / 1080;  // 矩形框从顶部开始
//        int rectRight = right * viewWidth / 1080;  // 矩形框到底部结束

        // 在画布上绘制矩形框
        canvas.drawRect(rectLeft, rectTop, rectRight, rectBottom, paint);
    }


    // Setter 方法，用于动态更新矩形位置
    public void setRectangle(int top, int bottom, int left, int right) {
        this.top = top;
        this.bottom = bottom;
        this.left = left;
        this.right = right;
        invalidate(); // 调用重绘方法
    }
}

