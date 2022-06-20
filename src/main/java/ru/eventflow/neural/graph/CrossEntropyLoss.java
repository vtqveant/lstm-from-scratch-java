package ru.eventflow.neural.graph;


import ru.eventflow.neural.Batch;

public class CrossEntropyLoss extends BaseNode {

    private Node y;
    private Node y_hat;
    private int vectorSize;

    public CrossEntropyLoss(int[] shape, Node y, Node y_hat) {
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

        Batch m_hat = y_hat.getValue();
        Batch m = y.getValue();
        Batch partial = new Batch(y.shape());

        int batchSize = y.getSize();

        // result is an average over all batch entries
        double loss = 0;
        for (int j = 0; j < batchSize; j++) {
            for (int i = 0; i < vectorSize; i++) {
                int[] idx = new int[]{j, i, 0};
                double y_i = m.getDouble(idx);
                double yhat_i = m_hat.getDouble(idx);

                loss -= y_i * Math.log(yhat_i) + (1 - y_i) * Math.log(1 - yhat_i);

                double dd = -y_i / yhat_i + (1 - y_i) / (1 - yhat_i);
                partial.put(idx, dd / vectorSize);
            }
        }
        loss = loss / vectorSize / batchSize;

        partials.put(y_hat, partial);

        value = new Batch(shape);
        value.put(new int[]{0, 0, 0}, loss);
        return value;
    }

}
