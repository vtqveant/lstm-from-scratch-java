package ru.eventflow.neural;

import Jama.Matrix;
import org.junit.Test;
import ru.eventflow.neural.graph.CrossEntropyLoss;
import ru.eventflow.neural.graph.Node;
import ru.eventflow.neural.graph.Placeholder;
import ru.eventflow.neural.graph.Softmax;

import static junit.framework.TestCase.assertTrue;

public class CrossEntropyLossTest {

    private final int m = 5;

    @Test
    public void testPositive() {
        // gold value
        Placeholder y = new Placeholder(new int[]{1, m, 1});
        Batch batch = new Batch(new int[]{1, m, 1});
        double[] elements = new double[m];
        for (int i = 0; i < m; i++) {
            elements[i] = 1.0 / m;
        }
        batch.put(0, new Matrix(elements, m));
        y.setValue(batch);

        // another value
        Placeholder x = new Placeholder(new int[]{1, m, 1});
        x.setValue(Batch.zeros(new int[]{1, m, 1}));
        Node y_hat = new Softmax(new int[]{1, m, 1}, x);

        // the loss
        Node loss = new CrossEntropyLoss(new int[]{1, 1, 1}, y, y_hat);
        loss.getValue().print();

        double v = loss.getValue().get(new int[]{0, 0, 0});
        assertTrue(v >= 0);
    }
}
