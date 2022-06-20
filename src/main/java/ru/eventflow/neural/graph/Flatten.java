package ru.eventflow.neural.graph;

import Jama.Matrix;
import ru.eventflow.neural.Batch;

/**
 * Converts a matrix to a vector by concatenating columns
 * <p>
 * [1, n, m] -> [1, n * m, 1]
 */
public class Flatten extends BaseNode {

    private Node child;

    public Flatten(int[] shape, Node child) {
        super(shape, child);

        this.child = child;
        this.shape = new int[]{1, child.getRows() * child.getColumns(), 1};
    }

    @Override
    public Batch getValue() {
        if (value != null) {
            return value;
        }

        Batch value = new Batch(shape);
        for (int i = 0; i < child.getColumns(); i++) { // vector
            for (int j = 0; j < child.getRows(); j++) { // vector component
                value.put(new int[]{0, i * child.getRows() + j, 0}, child.getValue().get(new int[]{0, j, i}));
            }
        }

        return value;
    }

    /**
     * just reshape the partial from above
     */
    @Override
    public Batch applyChainRule(Node child) {
        Matrix dual = getDualValue().get(0);
        Batch t = new Batch(child.shape());
        for (int i = 0; i < child.getColumns(); i++) { // vector
            for (int j = 0; j < child.getRows(); j++) { // vector component
                t.put(new int[]{0, j, i}, dual.get(i * child.getRows() + j, 0));
            }
        }
        return t;
    }

}
