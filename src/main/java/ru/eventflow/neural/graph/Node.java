package ru.eventflow.neural.graph;

import ru.eventflow.neural.Batch;

import java.util.List;
import java.util.Map;

public interface Node {

    List<Node> getChildren();

    List<Node> getParents();

    void addParent(Node parent);

    Batch getValue();

    Batch getDualValue();

    Map<Node, Batch> getPartials();

    void reset();

    int[] shape();

    Batch applyChainRule(Node child);

    int getSize();

    int getRows();

    int getColumns();

    Node copy();

}
