package ru.eventflow.neural.graph;

import ru.eventflow.neural.Batch;

import java.io.Serializable;

/**
 * Network parameters for optimization with SGD.
 * If not initialized, they will be zero everywhere.
 * <p>
 * Shape:
 * <p>
 * [1, 1]    -- a scalar values
 * [1, n]    -- an n-dimensional vector
 * [n, m]    -- a matrix n-by-m
 */
public class Variable extends BaseNode implements Serializable {

    private String name;

    public Variable(String name, int[] shape, Batch value) {
        this(shape, value);
        this.name = name;
    }

    public Variable(int[] shape) {
        super(shape);
    }

    public Variable(int[] shape, Batch value) {
        super(shape);
        this.value = value;
    }

    @Override
    public Batch getValue() {
        if (value == null) {
            value = new Batch(shape);
        }
        return value;
    }

    public void setValue(Batch value) {
        this.value = value;
    }

    @Override
    public Variable copy() {
        return new Variable(name, shape, value);
    }

    @Override
    public void reset() {
        // noop
    }

    public String getName() {
        return name;
    }

    @Override
    public Batch applyChainRule(Node child) {
        return new Batch(child.shape());
    }

}
