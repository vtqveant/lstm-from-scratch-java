package ru.eventflow.neural.graph;


import ru.eventflow.neural.Batch;

/**
 * Frobenius norm is used as a score of a linkage matrix, therefore we expect a square matrix as input
 * <p>
 * Shape:
 * [n, n] -- a square matrix (n-by-n)
 * <p>
 * Out:
 * [1, 1] -- a scalar
 * <p>
 * Derivative:
 * [n, n] -- a square matrix (n-by-n)
 */
public class FrobeniusNorm extends BaseNode {

    private Node matrix;

    public FrobeniusNorm(int[] shape, Node matrix) {
        super(shape, matrix);
        if (matrix.getRows() != matrix.getColumns()) {
            throw new IllegalArgumentException("Matrix should be square");
        }
        this.matrix = matrix;
    }

    @Override
    public Batch getValue() {
        if (value != null) {
            return value;
        }

        Batch m = matrix.getValue();
        Batch partial = new Batch(matrix.shape());

        // result
        double sum = 0;
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getColumns(); j++) {
                double v = m.getDouble(new int[]{0, i, j});
                sum += v * v;
            }
        }
        double norm = Math.sqrt(sum);

        // derivative
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getColumns(); j++) {
                double v = m.getDouble(new int[]{0, i, j});
                partial.put(new int[]{0, i, j}, v / norm);
            }
        }
        partials.put(matrix, partial);

        value = new Batch(new int[]{1, 1, 1});
        value.put(new int[]{0, 0, 0}, norm);
        return value;
    }

    /**
     * short-circuiting here a little bit, since we know that the dual is scalar
     */
    @Override
    public Batch applyChainRule(Node child) {
        Batch dual = getDualValue();
        Batch partial = partials.get(child);
        return partial.times(dual.get(new int[]{0, 0, 0}));
    }
}
