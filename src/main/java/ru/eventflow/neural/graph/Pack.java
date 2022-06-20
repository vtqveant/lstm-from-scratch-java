package ru.eventflow.neural.graph;

import Jama.Matrix;
import ru.eventflow.neural.Batch;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Takes a list of vectors and packs them in a matrix of column vectors
 */
public class Pack extends BaseNode {

    private Map<Node, Integer> indices = new HashMap<>();

    public Pack(int[] shape, List<Node> children) {
        super(shape, children.toArray(new Node[0]));
    }

    public Pack(int[] shape, Node... children) {
        super(shape, children);
        this.shape = new int[]{1, children[0].getRows(), children.length};
    }

    @Override
    public Batch getValue() {
        if (value != null) {
            return value;
        }

        Batch value = new Batch(shape);
        for (int i = 0; i < children.size(); i++) {
            Node vector = children.get(i);
            Matrix m = vector.getValue().get(0).getMatrix(0, vector.getRows() - 1, 0, 0);
            value.get(0).setMatrix(0, vector.getRows() - 1, i, i, m);
            indices.put(vector, i);
        }

        return value;
    }

    /**
     * it does not compute anything, it simply returns the column of the pack's dual value that corresponds to the child
     */
    @Override
    public Batch applyChainRule(Node child) {
        Batch dual = getDualValue();
        Batch result = new Batch(child.shape());
        int i = indices.get(child);
        result.put(0, dual.get(0).getMatrix(0, child.getRows() - 1, i, i));
        return result;
    }
}
