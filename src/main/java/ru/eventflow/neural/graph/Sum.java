package ru.eventflow.neural.graph;

import ru.eventflow.neural.Batch;

import java.util.List;

public class Sum extends BaseNode {

    public Sum(int[] shape, List<Node> children) {
        super(shape, children.toArray(new Node[children.size()]));
    }

    public Sum(int[] shape, Node... children) {
        super(shape, children);
    }

    @Override
    public Batch getValue() {
        if (value != null) {
            return value;
        }

        value = new Batch(shape);
        for (Node child : children) {
            value = value.plus(child.getValue());
            partials.put(child, Batch.ones(child.shape()));
        }

        return value;
    }

    @Override
    public Batch applyChainRule(Node child) {
        Batch partial = partials.get(child);
        Batch dual = getDualValue();

        if (partial.getColumns() != dual.getColumns() || partial.getRows() != dual.getRows()) {
            throw new IllegalArgumentException("Shape mismatch");
        }
        return partial.mul(dual);
    }

}
