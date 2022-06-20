package ru.eventflow.neural;

import Jama.Matrix;
import org.junit.Before;
import org.junit.Test;
import ru.eventflow.neural.graph.HammingLoss;
import ru.eventflow.neural.graph.Node;
import ru.eventflow.neural.graph.Placeholder;
import ru.eventflow.neural.rnn.Parameters;

import java.util.Collections;

import static org.junit.Assert.assertEquals;

public class HammingLossTest {

    private final static int m = 5;
    private Parameters parameters;

    @Before
    public void setUp() {
        parameters = new Parameters(1, 1, Collections.emptyList());
    }

    /**
     * y     = [0, 0, 1, 0, 1]
     * y_hat = [1, 0, 0, 1, 1]
     * <p>
     * H = 3/5 = 0.6
     */
    @Test
    public void testExactValue() {
        // gold value
        Placeholder y = new Placeholder(new int[]{1, m, 1});
        Batch batch = new Batch(new int[]{1, m, 1});
        batch.put(0, new Matrix(new double[]{0, 0, 1, 0, 1}, m));
        y.setValue(batch);

        // prediction
        Placeholder y_hat = new Placeholder(new int[]{1, m, 1});
        Batch prediction = new Batch(new int[]{1, m, 1});
        prediction.put(0, new Matrix(new double[]{1, 0, 0, 1, 1}, m));
        y_hat.setValue(prediction);

        // the loss
        Node loss = new HammingLoss(new int[]{1, 1, 1}, y, y_hat, 0, parameters);
        loss.getValue().print();

        double v = loss.getValue().get(new int[]{0, 0, 0});
        assertEquals(0.6, v, 0.001);
    }

    /**
     * y     = [0, 0, 1, 0, 1]
     * y_hat = [0, 0, 1, 0, 1]
     * <p>
     * H = 0
     */
    @Test
    public void testOptimal() {
        // gold value
        Placeholder y = new Placeholder(new int[]{1, m, 1});
        Batch batch = new Batch(new int[]{1, m, 1});
        batch.put(0, new Matrix(new double[]{0, 0, 1, 0, 1}, m));
        y.setValue(batch);

        // prediction
        Placeholder y_hat = new Placeholder(new int[]{1, m, 1});
        Batch prediction = new Batch(new int[]{1, m, 1});
        prediction.put(0, new Matrix(new double[]{0, 0, 1, 0, 1}, m));
        y_hat.setValue(prediction);

        // the loss
        Node loss = new HammingLoss(new int[]{1, 1, 1}, y, y_hat, 0, parameters);
        loss.getValue().print();

        double v = loss.getValue().get(new int[]{0, 0, 0});
        assertEquals(0.0, v, 0.001);
    }


}
