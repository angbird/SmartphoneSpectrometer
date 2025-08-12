package com.example.watermonitor;

import android.graphics.Color;
import android.view.View;

import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.components.XAxis;
import com.github.mikephil.charting.components.YAxis;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;

import java.util.ArrayList;
import java.util.List;

public class ChartHelper {

    private LineChart lineChart;

    public ChartHelper(LineChart lineChart) {
        this.lineChart = lineChart;
    }

    public void displayGraph(float[] data, String label, float yMin, float yMax, float xStart) {
        List<Entry> entries = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            entries.add(new Entry(xStart + i, data[i]));
        }

        LineDataSet dataSet = new LineDataSet(entries, label);
        dataSet.setColor(Color.BLUE);
        dataSet.setValueTextColor(Color.BLACK);

        LineData lineData = new LineData(dataSet);
        lineChart.setData(lineData);

        // 设置纵坐标的范围
        YAxis leftAxis = lineChart.getAxisLeft();
        leftAxis.setAxisMinimum(yMin);
        leftAxis.setAxisMaximum(yMax);

        YAxis rightAxis = lineChart.getAxisRight();
        rightAxis.setAxisMinimum(yMin);
        rightAxis.setAxisMaximum(yMax);

        // 设置横坐标轴格式
        XAxis xAxis = lineChart.getXAxis();
        xAxis.setPosition(XAxis.XAxisPosition.BOTTOM);
        xAxis.setGranularity(1f);
        xAxis.setAxisMinimum(xStart);
        xAxis.setAxisMaximum(xStart + data.length - 1);

        lineChart.invalidate(); // refresh
        lineChart.setVisibility(View.VISIBLE);
    }
}
