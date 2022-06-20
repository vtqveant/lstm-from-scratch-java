package ru.eventflow.neural.optimization;

import ru.eventflow.neural.Batch;
import ru.eventflow.neural.graph.Node;
import ru.eventflow.neural.graph.Variable;

import java.util.HashSet;
import java.util.Set;

public class SimpleOptimization implements Optimization {

    private double learningRate;
    private double threshold;
    private Node loss;
    private Set<Variable> variables = new HashSet<>();
    private Set<Node> nodes = new HashSet<>();

    public SimpleOptimization(double learningRate, double threshold, Node loss) {
        this.learningRate = learningRate;
        this.threshold = threshold;
        this.loss = loss;

        populateCollections(loss);
    }

    private void populateCollections(Node node) {
        if (!nodes.contains(node)) {
            nodes.add(node);
            if (node instanceof Variable && !variables.contains(node)) {
                variables.add((Variable) node);
            }
            for (Node n : node.getChildren()) {
                populateCollections(n);
            }
        }
    }

    private void reset() {
        for (Node node : nodes) {
            node.reset();
        }
    }

    @Override
    public void fit() {

        System.out.println("Training started");

        double loss_old = loss.getValue().get(new int[]{0, 0, 0});
        double loss_new = loss_old - 0.1; // pick another point to begin with approximation

        int i = 0;
        while (Math.abs(loss_old - loss_new) > threshold) {
//        while (loss_new > threshold || loss_new > 0) {
            long start_ts = System.currentTimeMillis();
            i++;
            loss_old = loss_new;

            // forward pass
            reset();
            loss_new = loss.getValue().get(new int[]{0, 0, 0});

            if (i % 100 == 0) {
//            if (true) {
                System.out.println("iteration = " + i + ", loss = " + loss_new);
            }

            // backward pass and gradients
            for (Variable variable : variables) {
                Batch gradient = variable.getDualValue();
                Batch update = gradient.times(-learningRate);
                variable.setValue(variable.getValue().plus(update));
            }

            long end_ts = System.currentTimeMillis();
//            System.out.println("iteration = " + i + " took " + (end_ts - start_ts) + " ms");
        }

        System.out.println();
        System.out.println("Total iterations = " + i);
    }

}
