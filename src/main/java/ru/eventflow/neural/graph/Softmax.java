package ru.eventflow.neural.graph;


import ru.eventflow.neural.Batch;

/**
 * Shape:
 * <p>
 * [k, n]    -- k n-dimensional vectors
 * <p>
 * Shape of derivative
 * <p>
 * [k, n, n] -- k n-by-n matrices
 * <p>
 * Takes k n-dimensional vectors and produces k n-dimensional vectors
 */
public class Softmax extends BaseNode {

    private Node child;
    private int batchSize;
    private int vectorSize;

    public Softmax(int[] shape, Node child) {
        super(shape, child);
        this.child = child;
        this.batchSize = child.getSize();
        this.vectorSize = child.getRows();
    }

    /**
     * s. http://cs231n.github.io/linear-classify/#softmax
     *
     * TODO apply log-sum-exp trick to avoid overflows
     */
    private static double[] softmax(double[] row) {
        double max = row[0];
        for (double d : row) {
            max = Math.max(max, d);
        }

        double denominator = 0;
        double[] values = new double[row.length];
        for (int i = 0; i < values.length; i++) {
            values[i] = row[i] - max;
            denominator += Math.exp(values[i]);
        }
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.exp(values[i]) / denominator;
        }
        return values;
    }

    @Override
    public Batch getValue() {
        if (value != null) {
            return value;
        }

        Batch partial = Batch.zeros(new int[]{batchSize, vectorSize, vectorSize});

        Batch in = child.getValue();
        value = Batch.manyVectors(batchSize, vectorSize);

        for (int batch = 0; batch < batchSize; batch++) {
            double[] values = softmax(in.get(batch).getColumnPackedCopy());
            for (int i = 0; i < vectorSize; i++) {
                value.put(new int[]{batch, i, 0}, values[i]);

                // derivatives
                for (int j = 0; j < vectorSize; j++) {
                    if (i == j) {
                        partial.put(new int[]{batch, j, j}, values[j] * (1 - values[j]));
                    } else {
                        partial.put(new int[]{batch, j, i}, -values[i] * values[j]);
                    }
                }
            }
        }
        partials.put(child, partial);

        return value;
    }

}
