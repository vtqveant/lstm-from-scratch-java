package ru.eventflow.neural.visualization;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;

public class AttentionVisualizer {

    /**
     * where to save generated files
     */
    private File directory;

    public AttentionVisualizer(File directory) {
        this.directory = directory;
    }

    private Icon buildIcon(double[][] data, double low, double high, String[] xValues, String[] yValues) {
        HeatChart map = buildHeatChart(data, low, high, xValues, yValues);
        Image image = map.getChartImage();
        return new ImageIcon(image);
    }

    public void saveToFile(double[][] data, double low, double high, String[] xValues, String[] yValues) throws IOException {
        HeatChart map = buildHeatChart(data, low, high, xValues, yValues);
        map.saveToFile(new File(directory, String.valueOf(System.currentTimeMillis()) + ".png"));
    }

    private HeatChart buildHeatChart(double[][] data, double low, double high, String[] xValues, String[] yValues) {
        HeatChart map = new HeatChart(data, low, high);
        map.setTitle("Attention matrix");
        map.setCellSize(new Dimension(40, 40));
        map.setLowValueColour(Color.BLACK);
        map.setHighValueColour(Color.WHITE);
        map.setXValuesHorizontal(true);
        map.setXValues(xValues);
        map.setAxisThickness(0);
        map.setYValues(yValues);
        map.setChartMargin(60);
        return map;
    }

}
