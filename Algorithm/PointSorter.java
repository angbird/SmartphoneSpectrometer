package com.example.watermonitor;



import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.List;

public class PointSorter {

    // 封装的重心排序函数，从大到小按角度排序
    public static List<Point> sortPointsByAngleDescending(List<Point> points) {
        // 计算重心
        Point centroid = new Point(0, 0);
        for (Point p : points) {
            centroid.x += p.x;
            centroid.y += p.y;
        }
        centroid.x /= points.size();
        centroid.y /= points.size();

        // 计算每个点的角度并排序
        List<PointWithAngle> pointsWithAngles = new ArrayList<>();
        for (Point p : points) {
            double angle = Math.toDegrees(Math.atan2(p.y - centroid.y, p.x - centroid.x));
            if (angle < 0) {
                angle += 360;  // 将角度范围调整为 0 到 360
            }
            pointsWithAngles.add(new PointWithAngle(p, angle));
        }

        // 按角度从大到小排序
        pointsWithAngles.sort((p1, p2) -> Double.compare(p2.angle, p1.angle));

        // 提取排序后的点
        List<Point> sortedPoints = new ArrayList<>();
        for (PointWithAngle p : pointsWithAngles) {
            sortedPoints.add(p.point);
        }

        return sortedPoints;
    }

    // 内部类用于存储点和角度
    private static class PointWithAngle {
        Point point;
        double angle;

        public PointWithAngle(Point point, double angle) {
            this.point = point;
            this.angle = angle;
        }
    }
}
