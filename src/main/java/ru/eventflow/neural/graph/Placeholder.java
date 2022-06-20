package ru.eventflow.neural.graph;


import ru.eventflow.neural.Batch;

/**
 * Input values for the computation.
 * <p>
 * Shape:
 * <p>
 * [1, 1]    -- a scalar values
 * [1, n]    -- an n-dimensional vector
 * [n, m]    -- a matrix n-by-m
 */
public class Placeholder extends BaseNode {

    private Batch value;

    public Placeholder(int[] shape) {
        super(shape);
    }

    @Override
    public Batch getValue() {
        return value;
    }

    public void setValue(Batch value) {
        this.value = value;
    }

    @Override
    public Batch applyChainRule(Node child) {
        return new Batch(child.shape());
    }

    @Override
    public void reset() {
        // noop
    }
}
