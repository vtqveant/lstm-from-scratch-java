package ru.eventflow.neural.graph;

import Jama.Matrix;
import ru.eventflow.neural.Batch;

import java.util.HashMap;
import java.util.Map;

/**
 * Конкатенация двух векторов,
 * Градиент, входящий на обратном пути, просто разбивается на части по длине конкатенируемых векторов в порядке,
 * в котором они конкатенировались
 *
 * Пока только для векторов
 */
public class Concat extends BaseNode {

    private Map<Node, Integer> indices = new HashMap<>();

    public Concat(int[] shape, Node... children) {
        super(shape, children);
    }

    @Override
    public Batch getValue() {
        if (value != null) {
            return value;
        }

        int totalLength = 0;
        for (Node child : children) {
            indices.put(child, totalLength); // starting index of the vector being concatenated
            totalLength += child.getRows();
        }

        Batch value = new Batch(new int[]{1, totalLength, 1});
        for (Node child : children) {
            int offset = indices.get(child);
            value.get(0).setMatrix(offset, offset + child.getRows() - 1, 0, 0, child.getValue().get(0));
        }

        return value;
    }

    /**
     * for each child simply use a corresponding portion of the gradient, i.e. split the dual
     */
    @Override
    public Batch applyChainRule(Node child) {
        Batch dual = getDualValue();
        Matrix m = dual.get(0);
        int offset = indices.get(child);
        Matrix portion = m.getMatrix(offset, offset + child.getRows() - 1, 0, dual.getColumns() - 1);
        Batch dualSplit = new Batch(new int[]{1, child.getRows(), dual.getColumns()});
        dualSplit.put(0, portion);
        return dualSplit;
    }
}
