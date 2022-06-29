package ru.eventflow.neural;

import Jama.Matrix;

import java.io.Serializable;

/**
 * All shapes are 3d. Column vectors, thus the usual order of matrices in the product.
 */
public class Batch implements Serializable {

    private int size;
    private int rows;
    private int columns;
    private Matrix[] values;

    public Batch(int[] shape) {
        this.size = shape[0];
        this.rows = shape[1];
        this.columns = shape[2];
        this.values = new Matrix[size];

        // TODO externalize initializer
        for (int i = 0; i < size; i++) {
            values[i] = new Matrix(rows, columns);
        }
    }

    public Batch(int size, int rows, int columns) {
        this(new int[]{size, rows, columns});
    }

    public static Batch scalar(double value) {
        Batch result = new Batch(new int[]{1, 1, 1});
        result.put(new int[]{0, 0, 0}, value);
        return result;
    }

    /**
     * A column vector
     */
    public static Batch oneVector(int vectorSize) {
        return new Batch(1, vectorSize, 1);
    }

    public static Batch manyVectors(int batchSize, int vectorSize) {
        return new Batch(batchSize, vectorSize, 1);
    }

    public static Batch oneMatrix(int rows, int columns) {
        return new Batch(1, rows, columns);
    }

    public static Batch manyMatrices(int batchSize, int rows, int columns) {
        return new Batch(batchSize, rows, columns);
    }

    public static Batch zeros(int[] shape) {
        return new Batch(shape);
    }

    public static Batch ones(int[] shape) {
        Batch batch = new Batch(shape);
        Matrix m = new Matrix(batch.rows, batch.columns);
        for (int j = 0; j < batch.rows; j++) {
            for (int k = 0; k < batch.columns; k++) {
                m.set(j, k, 1.0);
            }
        }
        for (int i = 0; i < batch.size; i++) {
            batch.put(i, m.copy());
        }
        return batch;
    }

    public static Batch rand(int[] shape) {
        Batch batch = new Batch(shape);
        for (int i = 0; i < shape[0]; i++) {
            batch.put(i, Matrix.random(shape[1], shape[2]));
        }
        return batch;
    }

    public static Batch eye(int[] shape) {
        Batch batch = new Batch(shape);
        for (int i = 0; i < batch.size; i++) {
            batch.put(i, Matrix.identity(batch.rows, batch.columns));
        }
        return batch;
    }

    public static Batch uniformDistributionVector(int vectorSize) {
        Batch batch = new Batch(new int[]{1, vectorSize, 1});
        double[] elements = new double[vectorSize];
        for (int i = 0; i < vectorSize; i++) {
            elements[i] = 1d / vectorSize;
        }
        batch.put(0, new Matrix(elements, vectorSize));
        return batch;
    }

    /**
     * Xavier initialization
     */
    public static Batch xavier(int[] shape) {
        Batch batch = new Batch(shape);
        double factor = Math.sqrt(6) / Math.sqrt(batch.rows + batch.columns);
        for (int i = 0; i < batch.size; i++) {
            Matrix m = new Matrix(batch.rows, batch.columns);
            for (int j = 0; j < batch.rows; j++) {
                for (int k = 0; k < batch.columns; k++) {
                    m.set(j, k, 2 * Math.random() * factor - factor);
                }
            }
            batch.put(i, m);
        }
        return batch;
    }

    public double getDouble(int[] index) {
        return values[index[0]].get(index[1], index[2]);
    }

    public int[] shape() {
        return new int[]{size, rows, columns};
    }

    public void put(int i, Matrix matrix) {
        if (i < 0 || i > size - 1) {
            throw new IllegalArgumentException("Invalid position");
        }
        if (matrix.getRowDimension() != rows || matrix.getColumnDimension() != columns) {
            throw new IllegalArgumentException("Shape mismatch");
        }
        values[i] = matrix;
    }

    public Matrix get(int i) {
        return values[i];
    }

    public void set(int[] index, double value) {
        get(index[0]).set(index[1], index[2], value);
    }

    public Batch plus(Batch other) {
        if (size != other.size) {
            throw new IllegalArgumentException("Shape mismatch");
        }
        Batch result = new Batch(size, rows, other.columns);
        for (int i = 0; i < size; i++) {
            try {
                result.put(i, get(i).plus(other.get(i)));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return result;
    }

    public Batch times(Batch other) {
        if (size != other.size || columns != other.rows) {
            throw new IllegalArgumentException("Shape mismatch");
        }
        Batch result = new Batch(size, rows, other.columns);
        for (int i = 0; i < size; i++) {
            result.put(i, get(i).times(other.get(i)));
        }
        return result;
    }

    public Batch times(double number) {
        Batch result = new Batch(size, rows, columns);
        for (int i = 0; i < size; i++) {
            result.put(i, get(i).times(number));
        }
        return result;
    }

    /**
     * element-wise multiplication, shapes must coincide
     */
    public Batch mul(Batch other) {
        Batch result = new Batch(size, rows, columns);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < rows; j++) {
                for (int k = 0; k < columns; k++) {
                    int[] index = new int[]{i, j, k};
                    result.put(index, get(index) * other.get(index));
                }
            }
        }
        return result;
    }

    /**
     * gradient clipping applied to each vector separately
     */
    public Batch clip(double threshold) {
        Batch result = new Batch(size, rows, columns);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < columns; j++) {
                Matrix vector = get(i).getMatrix(0, rows - 1, j, j);
                double norm = vector.norm2();
                if (norm >= threshold) {
                    result.get(i).setMatrix(0, rows - 1, j, j, vector.times(threshold / norm));
                } else {
                    result.get(i).setMatrix(0, rows - 1, j, j, vector);
                }
            }
        }
        return result;
    }

    public Batch transpose() {
        Batch result = new Batch(size, columns, rows);
        for (int i = 0; i < size; i++) {
            result.put(i, get(i).transpose());
        }
        return result;
    }

    public Batch inverse() {
        Batch result = new Batch(size, rows, columns);
        for (int i = 0; i < size; i++) {
            result.put(i, get(i).inverse());
        }
        return result;
    }

    public void put(int[] index, double value) {
        values[index[0]].set(index[1], index[2], value);
    }

    public double get(int[] index) {
        return values[index[0]].get(index[1], index[2]);
    }

    public void print() {
        print("");
    }

    public void print(String message) {
        System.out.println(message + " [" + size + ", " + rows + ", " + columns + "]");
        for (int i = 0; i < size; i++) {
            values[i].print(4, 6); // formatting
        }
    }

    public int getSize() {
        return size;
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    public Batch copy() {
        Batch copy = new Batch(shape());
        for (int i = 0; i < size; i++) {
            copy.put(i, get(i).copy());
        }
        return copy;
    }

    @Override
    public String toString() {
        return "[" + size + ", " + rows + ", " + columns + "]";
    }
}
