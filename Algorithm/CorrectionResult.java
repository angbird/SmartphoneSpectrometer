package com.example.watermonitor;

import android.graphics.Bitmap;
import org.opencv.core.Point;
import java.util.List;

public class CorrectionResult {
    private Bitmap alignedBitmap;
    private List<Point> orderedPts2;
    private double rotatedAngle;

    public CorrectionResult(Bitmap alignedBitmap, List<Point> orderedPts2, double rotatedAngle) {
        this.alignedBitmap = alignedBitmap;
        this.orderedPts2 = orderedPts2;
        this.rotatedAngle = rotatedAngle;
    }

    public Bitmap getAlignedBitmap() {

        return alignedBitmap;
    }

    public List<Point> getOrderedPts2() {
        return orderedPts2;
    }

    public double getRotatedAngle() {
        return rotatedAngle;
    }
}
