package ru.eventflow.neural.graph;

import ru.eventflow.neural.Batch;

/**
 * this implementation only applies to vectors
 */
public class Sigmoid extends BaseNode {

    private Node child;
    private int vectorSize;

    public Sigmoid(int[] shape, Node child) {
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
                double v = 1d / (1 + Math.exp(-in.get(new int[]{i, j, 0})));
                value.put(new int[]{i, j, 0}, v);
                partial.put(new int[]{i, j, j}, v * (1 - v));
            }
        }
        partials.put(child, partial);

        return value;
    }

}
