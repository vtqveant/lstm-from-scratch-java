package ru.eventflow.neural.graph;

import ru.eventflow.neural.Batch;

public class Involution extends BaseNode {

    private Node fact;
    private Node antiphases;

    public Involution(int[] shape, Node fact, Node antiphases) {
        super(shape, fact, antiphases);

        this.fact = fact;
        this.antiphases = antiphases;
    }

    @Override
    public Batch getValue() {
        if (value != null) {
            return value;
        }

        int n = antiphases.getRows();

        value = new Batch(new int[]{1, n, 1});
        Batch partial = new Batch(fact.shape());
        for (int i = 0; i < n; i++) {
            int[] index = new int[]{0, i, 0};
            value.set(index, antiphases.getValue().get(index) - fact.getValue().get(index));
            partial.set(index, -1.0);
        }
        partials.put(fact, partial);
        return value;
    }

    @Override
    public Batch applyChainRule(Node child) {
        Batch partial = partials.get(child);
        Batch dual = getDualValue();

        if (partial == null) {
            System.err.println("partial is null");
        }

        if (partial.getColumns() != dual.getColumns() || partial.getRows() != dual.getRows()) {
            throw new IllegalArgumentException("Shape mismatch");
        }
        return partial.mul(dual);
    }

}
