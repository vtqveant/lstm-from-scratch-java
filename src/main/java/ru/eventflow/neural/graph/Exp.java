package ru.eventflow.neural.graph;

import ru.eventflow.neural.Batch;

// TODO test
public class Exp extends BaseNode {

    private Node child;

    public Exp(int[] shape, Node child) {
        super(shape, child);
        this.child = child;
    }

    @Override
    public Batch getValue() {
        if (value != null) {
            return value;
        }

        Batch in = child.getValue();

        int batchSize = child.shape()[0];
        int vectorSize = child.shape()[1];
        int numVectors = child.shape()[2];

        value = Batch.zeros(child.shape());
        Batch partial = Batch.zeros(new int[]{batchSize, vectorSize, numVectors});
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < vectorSize; j++) {
                for (int k = 0; k < numVectors; k++) {
                    double v = Math.exp(in.get(new int[]{i, j, k}));
                    value.put(new int[]{i, j, k}, v);
                    partial.put(new int[]{i, j, k}, v);
                }
            }
        }
        partials.put(child, partial);

        return value;
    }

}
