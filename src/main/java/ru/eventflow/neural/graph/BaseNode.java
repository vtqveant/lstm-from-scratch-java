package ru.eventflow.neural.graph;

import ru.eventflow.neural.Batch;

import java.util.*;

public abstract class BaseNode implements Node {

    List<Node> children;
    List<Node> parents;
    Map<Node, Batch> partials;
    int[] shape; // like in ND4j
    Batch value;
    Batch dualValue;

    BaseNode(int[] shape, Node... children) {
        this.shape = shape;
        this.children = Arrays.asList(children);
        for (Node child : children) {
            child.addParent(this);
        }
        this.parents = new ArrayList<>();
        this.partials = new HashMap<>();
    }

    @Override
    public List<Node> getChildren() {
        return children;
    }

    @Override
    public List<Node> getParents() {
        return parents;
    }

    @Override
    public int[] shape() {
        return shape;
    }

    @Override
    public void addParent(Node parent) {
        parents.add(parent);
    }

    public void reset() {
        value = null;
        dualValue = null;
        partials.clear();
    }

    @Override
    public Map<Node, Batch> getPartials() {
        return partials;
    }

    @Override
    public abstract Batch getValue();

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
                dualValue = dualValue.plus(b);
            }
        }

        return dualValue;
    }

    /**
     * batch size, 3d tensor depth
     */
    @Override
    public int getSize() {
        return shape[0];
    }

    @Override
    public int getRows() {
        return shape[1];
    }

    @Override
    public int getColumns() {
        return shape[2];
    }

    @Override
    public Batch applyChainRule(Node child) {
        Batch dual = getDualValue();
        Batch partial = partials.get(child);
        return partial.times(dual);
    }

    public Node copy() {
        throw new AssertionError("Not implemented");
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(getClass().getSimpleName());
        sb.append(" [");
        for (int i = 0; i < shape.length; i++) {
            sb.append(shape[i]);
            if (i < shape.length - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        BaseNode baseNode = (BaseNode) o;
        return Objects.equals(children, baseNode.children);
    }

    @Override
    public int hashCode() {
        return Objects.hash(children);
    }
}
