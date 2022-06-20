package ru.eventflow.neural.graph;


import ru.eventflow.neural.Batch;

/**
 * Shape:
 * <p>
 * [k, n, 1] x [k, n, 1] -- two batches of size k of n-dimensional vectors
 * <p>
 * Result:
 * [1, 1, 1]
 * <p>
 * Derivative:
 * [k, n, 1]
 */
public class MSELoss extends BaseNode {

    private Node y;
    private Node y_hat;
    private int vectorSize;

    public MSELoss(int[] shape, Node y, Node y_hat) {
        super(shape, y, y_hat);
        if (y.getSize() != y_hat.getSize() || y.getRows() != y_hat.getRows()) {
            throw new IllegalArgumentException("Shape mismatch");
        }
        this.y = y;
        this.y_hat = y_hat;
        this.vectorSize = y.getRows();
    }

    @Override
    public Batch getValue() {
        if (value != null) {
            return value;
        }

        Batch m = y.getValue();
        Batch m_hat = y_hat.getValue();

        int batchSize = y.getSize();
        Batch partial = new Batch(y.shape());

        // result is an average over all batch entries
        double loss = 0;
        for (int j = 0; j < batchSize; j++) {
            for (int i = 0; i < vectorSize; i++) {
                int[] idx = new int[]{j, i, 0};

                loss += (m.getDouble(idx) - m_hat.getDouble(idx)) * (m.getDouble(idx) - m_hat.getDouble(idx));

                double ddy_i = 2 * (m_hat.getDouble(idx) * m_hat.getDouble(idx) - m.getDouble(idx) * m_hat.getDouble(idx)) / vectorSize;
                partial.put(idx, ddy_i);
            }
        }
        loss = loss / vectorSize / batchSize;

        partials.put(y_hat, partial);

        value = Batch.oneVector(1);
        value.put(new int[]{0, 0, 0}, loss);
        return value;
    }

}
