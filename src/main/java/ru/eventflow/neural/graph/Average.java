package ru.eventflow.neural.graph;

import ru.eventflow.neural.Batch;

import java.util.List;

/**
 * s. https://stats.stackexchange.com/questions/183840/sum-or-average-of-gradients-in-mini-batch-gradient-decent
 */
public class Average extends BaseNode {

    public Average(int[] shape, List<Node> children) {
        super(shape, children.toArray(new Node[children.size()]));
    }

    public Average(int[] shape, Node... children) {
        super(shape, children);
    }

    @Override
    public Batch getValue() {
        if (value != null) {
            return value;
        }

        int n = children.size();
        value = new Batch(shape);
        for (Node child : children) {
            value = value.plus(child.getValue());
            partials.put(child, Batch.ones(child.shape()).times(1d / n));
        }
        value = value.times(1d / n);

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
