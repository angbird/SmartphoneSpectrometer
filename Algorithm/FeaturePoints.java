package com.example.watermonitor;

import org.opencv.core.Point;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class FeaturePoints {
    private static final int MAX_SIZE = 15; // 最大存储行数
    private LinkedList<List<Point>> pointsHistory;

    public FeaturePoints() {
        pointsHistory = new LinkedList<>();
    }

    public void addPoints(List<Point> points) {
        if (points.size() != 4) {
            throw new IllegalArgumentException("每行必须包含4个点");
        }

        if (pointsHistory.size() >= MAX_SIZE) {
            pointsHistory.pollFirst(); // 移除最早的行（先进先出）
        }
        pointsHistory.addLast(new ArrayList<>(points)); // 添加新的行
    }

    public List<List<Point>> getPointsHistory() {
        return new ArrayList<>(pointsHistory); // 返回所有历史数据
    }

    public List<Point> getLastPoints() {
        return pointsHistory.isEmpty() ? null : pointsHistory.getLast(); // 获取最新的行
    }

    // 计算每列的平均值
    public List<Point> getAveragePoints() {
        if (pointsHistory.isEmpty()) {
            return new ArrayList<>(); // 如果没有历史数据，返回空列表
        }

        // 初始化存储4列平均值的点列表
        List<Point> averagePoints = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            double sumX = 0;
            double sumY = 0;
            int count = 0;

            // 遍历历史记录的每一行
            for (List<Point> row : pointsHistory) {
                Point pt = row.get(i); // 获取当前列的点
                sumX += pt.x;
                sumY += pt.y;
                count++;
            }

            // 计算当前列的平均值
            averagePoints.add(new Point(sumX / count, sumY / count));
        }

        return averagePoints; // 返回包含4个平均点的列表
    }
}
