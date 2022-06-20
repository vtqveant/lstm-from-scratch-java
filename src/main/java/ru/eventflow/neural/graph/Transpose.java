package ru.eventflow.neural.graph;

import ru.eventflow.neural.Batch;

/**
 * Any operation which merely rearranges the elements of X, (e.g. vec, transpose, reshape, vecpose, block-vecpose)
 * rearranges the elements of dX in the same way.
 *
 * This implementation only applies to matrices, i.e. to [1, n, m] shapes.
 */
public class Transpose extends BaseNode {

    private final Node child;

    public Transpose(int[] shape, Node child) {
        super(shape, child);
        this.child = child;
    }

    @Override
    public Batch getValue() {
        if (value != null) {
            return value;
        }

        Batch value = new Batch(shape);
        value.put(0, child.getValue().get(0).transpose());

        return value;
    }

    @Override
    public Batch getDualValue() {
        if (dualValue != null) {
            return dualValue;
        }

        if (parents.size() == 0) {
            dualValue = Batch.ones(shape);
        } else {
            dualValue = Batch.zeros(shape);
            for (Node parent : parents) {
                Batch b = parent.applyChainRule(this);
                Batch t = new Batch(shape);
                // TODO ? if the parent is Matmul, we have this (probably we should have called parent.applyChainRule() with the transpose of `this`, but I kant)
                if (t.getRows() == b.getRows() && t.getColumns() == b.getColumns()) {
                    t.put(0, b.get(0));
                } else {
                    t.put(0, b.get(0).transpose());
                }
                dualValue = dualValue.plus(t);
            }
        }

        return dualValue;
    }

    @Override
    public Batch applyChainRule(Node child) {
        Batch t = new Batch(child.shape());
        t.put(0, getDualValue().get(0).transpose());
        return t;
    }
}
