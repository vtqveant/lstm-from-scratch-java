package ru.eventflow.neural.graph;

import ru.eventflow.neural.Batch;

/**
 * applies only to vectors
 */
public class ReLU extends BaseNode {

    private Node child;
    private int vectorSize;

    public ReLU(int[] shape, Node child) {
        super(shape, child);
        this.child = child;
        this.vectorSize = child.shape()[1];
    }

    @Override
    public Batch getValue() {
        if (value != null) {
            return value;
        }

        Batch in = child.getValue();

        int batchSize = child.shape()[0];

        value = Batch.zeros(child.shape());
        Batch partial = Batch.zeros(new int[]{batchSize, vectorSize, vectorSize});
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < vectorSize; j++) {
                double v = in.get(i).get(j, 0);
                if (v > 0) {
                    value.put(new int[]{i, j, 0}, v);
                    partial.put(new int[]{i, j, j}, 1.0);
                }
            }
        }
        partials.put(child, partial);

        return value;
    }

}
