package ru.eventflow.neural.graph;


import ru.eventflow.neural.Batch;

/**
 * Hadamard (element-wise) product.
 * <p>
 * Shape:
 * <p>
 * [k, 1]    -- k scalar values
 * [k, n]    -- k n-dimensional vectors
 * [k, n, m] -- k matrices n-by-m
 */
public class Mul extends BaseNode {

    private Node left;
    private Node right;

    public Mul(int[] shape, Node left, Node right) {
        super(shape, left, right);
        this.left = left;
        this.right = right;
    }

    @Override
    public Batch getValue() {
        if (value != null) {
            return value;
        }

        Batch m1 = left.getValue();
        Batch m2 = right.getValue();

        partials.put(left, m2);
        partials.put(right, m1);

        value = m1.mul(m2);
        return value;
    }

    @Override
    public Batch applyChainRule(Node child) {
        Batch dual = getDualValue();
        Batch partial = partials.get(child);
        return partial.mul(dual);
    }
}
