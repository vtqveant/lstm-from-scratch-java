package ru.eventflow.neural;

import Jama.Matrix;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class BatchMatMulTest {

    @Test
    public void testOneMatrixOneMatrix() {
        Batch a = Batch.oneMatrix(2, 3);
        a.put(0, new Matrix(new double[][]{
                new double[]{1, 0, 0},
                new double[]{0, 1, 0},
        }));
        a.print();

        Batch b = Batch.oneMatrix(3, 5);
        b.put(0, new Matrix(new double[][]{
                new double[]{1, 0, 0, 0, 0},
                new double[]{-1, 0, 0, 0, 0},
                new double[]{0, 0, 0, 0, 1},
        }));
        b.print();

        Batch result = a.times(b);
        result.print();

        assertEquals(1, result.getSize());
        assertEquals(2, result.getRows());
        assertEquals(5, result.getColumns());
    }

    @Test
    public void testOneMatrixOneVector() {
        Batch a = Batch.oneMatrix(3, 5);
        a.print();

        Batch b = Batch.oneVector(5);
        b.print();

        Batch result = a.times(b);
        result.print();

        assertEquals(1, result.getSize());
        assertEquals(3, result.getRows());
        assertEquals(1, result.getColumns());
    }

    @Test
    public void testTwoMatricesTwoVectors() {
        Batch a = Batch.manyMatrices(2, 3, 3);
        a.put(0, Matrix.identity(3, 3));
        a.put(1, Matrix.identity(3, 3).times(-1));
        a.print();

        Batch b = Batch.manyVectors(2, 3);
        b.put(0, new Matrix(new double[]{0, 1, 2}, 3)); // 3 rows, 1 column
        b.put(1, new Matrix(new double[]{1, 0, -1}, 3));
        b.print();

        Batch result = a.times(b);
        result.print();

        assertEquals(2, result.getSize());
        assertEquals(3, result.getRows());
        assertEquals(1, result.getColumns());
    }

    @Test
    public void testTwoMatricesTwoMatrices() {
        Batch a = Batch.manyMatrices(2, 3, 5);
        a.put(0, Matrix.identity(3, 5));
        a.print();

        Batch b = Batch.manyMatrices(2, 5, 2);
        b.put(0, Matrix.identity(5, 2));
        b.print();

        Batch result = a.times(b);
        result.print();

        assertEquals(2, result.getSize());
        assertEquals(3, result.getRows());
        assertEquals(2, result.getColumns());
    }

}
