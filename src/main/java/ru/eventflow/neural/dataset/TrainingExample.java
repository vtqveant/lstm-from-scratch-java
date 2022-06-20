package ru.eventflow.neural.dataset;

import java.util.List;


public class TrainingExample {

    private List<String> source;

    private List<String> target;

    public TrainingExample() {
    }

    public TrainingExample(List<String> source, List<String> target) {
        this.source = source;
        this.target = target;
    }

    public List<String> getSource() {
        return source;
    }

    public List<String> getTarget() {
        return target;
    }
}
